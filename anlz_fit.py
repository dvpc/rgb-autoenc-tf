
import numpy as N
import argparse, os

parser = argparse.ArgumentParser(prog='anlz.fit')
parser.add_argument('checkpoint_file')
parser.add_argument('-dbg', action='store_true', 
	help="print all tensor variables in checkpoint file. then exit.")
parser.add_argument('-name', 
	help="name of tensorvariable to process")
parser.add_argument('-b', '--begin', type=int, default=None)
parser.add_argument('-n', '--num', type=int, default=None)
parser.add_argument('-pk_file', default=None, 
	help='append existig file')
parser.add_argument('-silent', action='store_true', 
	help='no tmp output png is written')
parser.add_argument('-min', '--min_error_accept', type=float, default=0.03,
	help='min error to accept')
parser.add_argument('-zero', '--zero_tolerance', type=float, default=2e-01,
	help='tolerance of whether a RF vector is zero')

args = parser.parse_args()

'''check if ckpt file exists'''
if not os.path.exists(args.checkpoint_file):
	print 'file does not exist ', args.checkpoint_file
	exit()
abs_ckpt_file = os.path.abspath(args.checkpoint_file)
working_dir = str(os.path.split(abs_ckpt_file)[:-1][0])

'''read tf ckpt file and extract W'''
import tensorflow as tf 
try:
	reader = tf.train.NewCheckpointReader(args.checkpoint_file)
	'''print all tensor variables in checkpoint file. then exit.'''
	if args.dbg:
		print(reader.debug_string().decode("utf-8"))
		exit()
	'''retrieve tensor variable'''
	if args.name == None:
		tfvar_name = 'W_1'
	else:
		tfvar_name = args.name
	tfvar = reader.get_tensor(tfvar_name)
except Exception as e:
	raise e

print '___extracting variable:'
tfvar = N.swapaxes(tfvar, 0,1)
print tfvar_name, tfvar.shape
print 'max', N.max(tfvar), 'min', N.min(tfvar)
channel_dim = int(tfvar[0].shape[0]/3)
patch_dim = int(channel_dim**.5)
print 'ch_dim', channel_dim, 'ptch_dim', patch_dim
print 

'''check bounds'''
if args.begin is None:
	args.begin = 0
if args.num is None:
	args.num = tfvar.shape[0] - args.begin
print '___fitting #rf from', args.begin, 'to', str(args.begin+args.num)+':'
if args.begin+args.num > tfvar.shape[0]:
	print 'rf bounds larger than W!', \
		args.begin+args.num, '>=', tfvar.shape[0]
	print 'exiting'
	exit()
print 

'''make list of rf indexes'''
list_rf_idx = [i for i in xrange(args.begin,args.begin+args.num)]


'''fit the list'''
fit_results = []
for rf_idx in list_rf_idx:
	rf = tfvar[rf_idx]

	from anlz import is_close_to_zero
	if is_close_to_zero(rf, atol=args.zero_tolerance):
		print 'rf is zero! skipping...'
		fit_results.append( (rf_idx, [], []) )
		continue

	from anlz.fit import best_fit_wrapper
	result, c_col = best_fit_wrapper(
		rf, 
		channel_dim, 
		patch_dim, 
		num_fit_attempts=10,
		maxiter=1000,
		debug_idx=rf_idx,
		num_pool_threads=3,
		min_error_accept=args.min_error_accept
		)
	fit_results.append( (rf_idx, result, c_col) )

	if not args.silent:
		'''write temp result tilemap'''
		from anlz import write_rf_fit_map
		write_rf_fit_map(
			list_rf_idx, 
			fit_results, 
			patch_dim, 
			channel_dim, 
			tfvar, 
			filename=os.path.join(working_dir, 'tmp_rf_tilemap.png'))



'''make result dic'''
result_dic = {}
result_dic_fits = {}
'''check if results can be appended'''
if args.pk_file is None:
	pass
elif os.path.exists(args.pk_file):
	suffix = str(os.path.split(args.pk_file)[-1]).split('.')[-1]
	if suffix == 'cpkl':
		from anlz import deserialize_results
		result_dic = deserialize_results(
			filepath=args.pk_file)
		result_dic_fits = result_dic['fits']
'''fill new results into dic'''
for res in fit_results:
	key = res[0]
	result_dic_fits[key] = {
		'p': res[1],
		'c_color': res[2]
	}
result_dic['fits'] = result_dic_fits
result_dic['shape'] = tfvar.shape
result_dic['tfmax'] = N.max(tfvar)
result_dic['W'] = tfvar
'''write results'''
from anlz import serialize_results
serialize_results(
	filepath=os.path.join(working_dir, 'tmp_fits.cpkl'), 
	res_dict=result_dic)



'''write final debug tile map'''
from anlz import write_rf_fit_map
write_rf_fit_map(
	result_dic_fits.keys(),
	[(key, result_dic_fits[key]['p'], result_dic_fits[key]['c_color']) for key in result_dic_fits.keys()], 
	patch_dim, 
	channel_dim, 
	tfvar, 
	filename=os.path.join(working_dir, 'tmp_rf_tilemap_fin.png'))

# print result_dic.keys()
print len(result_dic_fits.keys())
# for key in result_dic.keys():
# 	print key 
# 	print result_dic[key]['p'] 
# 	print result_dic[key]['c_color']















