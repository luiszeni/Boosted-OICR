import logging
import numpy as np
import os
import shutil
import uuid
import subprocess

from tasks.config import cfg
from datasets.corloc import corloc_eval
from six.moves import cPickle as pickle

from pdb import set_trace as pause

logger = logging.getLogger(__name__)

def save_object(obj, file_name):
    """Save a Python object by pickling it."""
    file_name = os.path.abspath(file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def evaluate_boxes(voc_dataset, all_boxes, output_dir,  use_salt=True, cleanup=True, test_corloc=False, use_matlab=False):
    
    # print("args.use_matlab", use_matlab)
    
    salt = '_{}'.format(str(uuid.uuid4())) if use_salt else ''
    filenames = _write_voc_results_files(voc_dataset, all_boxes, salt)

    if test_corloc:
        _eval_discovery(voc_dataset, salt, output_dir)
    else:
        _do_matlab_eval(voc_dataset, salt,  all_boxes is not None, output_dir)
    
    if cleanup:
        if all_boxes is not None:
            for filename in filenames:
                shutil.copy(filename, output_dir)
                os.remove(filename)
    return None


def _write_voc_results_files(voc_dataset, all_boxes, salt):

    filenames = []
    filenames_cls = []

    
    for cls_ind, cls in enumerate(voc_dataset.class_labels):
        
        if cls == '__background__':
            continue

        logger.info('Writing VOC results for: {}'.format(cls))
        filename, filename_cls = _get_voc_results_file_template(voc_dataset, salt)

        filename = filename.format(cls)

     
        filenames.append(filename)
        if all_boxes is not None:
            # assert len(all_boxes[cls_ind + 1]) == len(image_index)
            
            with open(filename, 'wt') as f:

                for im_ind in range(len(voc_dataset)):

                    index =  voc_dataset.images[im_ind].split('/')[-1].replace(".jpg","")
                   
                    dets = all_boxes[cls_ind + 1][im_ind]
                    if type(dets) == list:
                        assert len(dets) == 0, \
                            'dets should be numpy.ndarray or empty list'
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

        
    
    return filenames


def _get_voc_results_file_template(voc_dataset, salt):
    
    image_set = voc_dataset.image_set
    year = voc_dataset.year  
    devkit_path = os.path.join(voc_dataset.root, 'VOCdevkit')
    
    filename     = 'comp4' + salt + '_det_' + image_set + '_{:s}.txt'
    filename_cls = 'comp1' + salt + '_cls_' + image_set + '_{:s}.txt'

    dirname = os.path.join(devkit_path, 'results', 'VOC' + year, 'Main')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    return os.path.join(dirname, filename), os.path.join(dirname, filename_cls)


def _eval_discovery(voc_dataset, salt, output_dir='output'):
    
    year = voc_dataset.year  
    image_set = voc_dataset.image_set
    devkit_path = os.path.join(voc_dataset.root, 'VOCdevkit')

    anno_path = os.path.join(devkit_path, 'VOC' + year, 'Annotations', '{:s}.xml')
    image_set_path = os.path.join(devkit_path, 'VOC' + year, 'ImageSets', 'Main', image_set + '.txt')

    cachedir = os.path.join(devkit_path, 'annotations_dis_cache_{}'.format(year))
    
    corlocs = []
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for _, cls in enumerate(voc_dataset.class_labels):
        if cls == '__background__':
            continue


        filename, filename_cls = _get_voc_results_file_template(voc_dataset, salt)

        filename = filename.format(cls)
        

        corloc = corloc_eval(filename, anno_path, image_set_path, cls, cachedir, ovthresh=0.5)
        corlocs += [corloc]
        logger.info('CorLoc for {} = {:.4f}'.format(cls, corloc))
        res_file = os.path.join(output_dir, cls + '_corloc.pkl')
        save_object({'corloc': corloc}, res_file)
    
    logger.info('Mean CorLoc = {:.4f}'.format(np.mean(corlocs)))
    logger.info('~~~~~~~~')
    logger.info('Results:')
    for corloc in corlocs:
        logger.info('{:.3f}'.format(corloc))
    logger.info('{:.3f}'.format(np.mean(corlocs)))
    logger.info('~~~~~~~~')



def _do_matlab_eval(voc_dataset, salt, eval_det = True, output_dir='output'):

    logger.info('-----------------------------------------------------------------------')
    logger.info('Computing results with the official MATLAB eval code adapted to octave.')
    logger.info('-----------------------------------------------------------------------')

    
    
    path = os.path.join(cfg.ROOT_DIR, 'code', 'datasets', 'VOCdevkit-matlab-wrapper')

    dev_kit_year = 2012
    dataset_year = 2007

    cmd = 'cd {} && '.format(path) + '{:s} --eval '.format(cfg.MATLAB)
    devkit_path = os.path.join(voc_dataset.root, 'VOCdevkit')

    output_path = os.path.join(cfg.ROOT_DIR, output_dir)

    cmd_det = cmd + '"voc_eval (\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d},{:d},{:s});"'.format(devkit_path, 'comp4' + salt, voc_dataset.image_set, output_path, dev_kit_year, dataset_year, 'false')


    logger.info('Running:\n{}'.format(cmd_det))
    subprocess.call(cmd_det, shell=True)
