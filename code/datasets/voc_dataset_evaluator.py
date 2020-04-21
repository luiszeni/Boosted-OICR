import logging
import numpy as np
import os
import shutil
import uuid
import subprocess

from tasks.config import cfg
from datasets.dataset_catalog import DATASETS
from datasets.dataset_catalog import DEVKIT_DIR
from datasets.corloc import corloc_eval
from six.moves import cPickle as pickle

from pdb import set_trace as pause

logger = logging.getLogger(__name__)

def save_object(obj, file_name):
    """Save a Python object by pickling it."""
    file_name = os.path.abspath(file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def evaluate_boxes(json_dataset, all_boxes, output_dir, all_cls_scores, use_salt=True, cleanup=True, test_corloc=False, use_matlab=False):
    
    # print("args.use_matlab", use_matlab)
    
    salt = '_{}'.format(str(uuid.uuid4())) if use_salt else ''
    filenames, filenames_cls = _write_voc_results_files(json_dataset, all_boxes, salt, all_cls_scores)
    
    if test_corloc:
        _eval_discovery(json_dataset, salt, output_dir)
    else:
        _do_matlab_eval(json_dataset, salt,  all_boxes is not None, output_dir)
    
    if cleanup:
        if all_boxes is not None:
            for filename in filenames:
                shutil.copy(filename, output_dir)
                os.remove(filename)

        for filename in filenames_cls:
            shutil.copy(filename, output_dir)
            os.remove(filename)
    return None


def _write_voc_results_files(json_dataset, all_boxes, salt, all_cls_scores):

    filenames = []
    filenames_cls = []
    image_set_path = voc_info(json_dataset)['image_set_path']
    
    assert os.path.exists(image_set_path), 'Image set path does not exist: {}'.format(image_set_path)
    
    with open(image_set_path, 'r') as f:
        image_index = [x.strip() for x in f.readlines()]
    
    # Sanity check that order of images in json dataset matches order in the
    # image set
    roidb = json_dataset.get_roidb()
    for i, entry in enumerate(roidb):
        index = os.path.splitext(os.path.split(entry['image'])[1])[0]
        assert index == image_index[i]
    
    for cls_ind, cls in enumerate(json_dataset.classes):
        
        if cls == '__background__':
            continue

        logger.info('Writing VOC results for: {}'.format(cls))
        filename, filename_cls = _get_voc_results_file_template(json_dataset, salt)

        filename     = filename.format(cls)
        filename_cls = filename_cls.format(cls)

        # print(filename_cls)

     
        filenames.append(filename)
        filenames_cls.append(filename_cls)
        if all_boxes is not None:
            assert len(all_boxes[cls_ind + 1]) == len(image_index)
            
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(image_index):
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

        with open(filename_cls, 'wt') as f:
            for im_ind, index in enumerate(image_index):
                if index not in all_cls_scores:
                    break
                clssy = all_cls_scores[index][0,cls_ind]
                f.write('{:s} {:.3f}\n'.format(index, clssy))
    
    return filenames, filenames_cls


def _get_voc_results_file_template(json_dataset, salt):
    info = voc_info(json_dataset)
    year = info['year']
    image_set = info['image_set']
    devkit_path = info['devkit_path']
    # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
    
    filename = 'comp4' + salt + '_det_' + image_set + '_{:s}.txt'
    filename_cls = 'comp1' + salt + '_cls_' + image_set + '_{:s}.txt'

    dirname = os.path.join(devkit_path, 'results', 'VOC' + year, 'Main')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return os.path.join(dirname, filename), os.path.join(dirname, filename_cls)


def _eval_discovery(json_dataset, salt, output_dir='output'):
    info = voc_info(json_dataset)
    year = info['year']
    anno_path = info['anno_path']
    image_set_path = info['image_set_path']
    devkit_path = info['devkit_path']
    cachedir = os.path.join(devkit_path, 'annotations_dis_cache_{}'.format(year))
    corlocs = []
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for _, cls in enumerate(json_dataset.classes):
        if cls == '__background__':
            continue
        # pause()
        filename = _get_voc_results_file_template(json_dataset, salt)[0].format(cls)
        corloc = corloc_eval(
            filename, anno_path, image_set_path, cls, cachedir, ovthresh=0.5)
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



def _do_matlab_eval(json_dataset, salt, eval_det = True, output_dir='output'):

    logger.info('-----------------------------------------------------------------------')
    logger.info('Computing results with the official MATLAB eval code adapted to octave.')
    logger.info('-----------------------------------------------------------------------')

    info = voc_info(json_dataset)
    
    path = os.path.join(cfg.ROOT_DIR, 'code', 'datasets', 'VOCdevkit-matlab-wrapper')

    dev_kit_year = 2012
    dataset_year = 2007

    cmd = 'cd {} && '.format(path) + '{:s} --eval '.format(cfg.MATLAB)

    output_path = os.path.join(cfg.ROOT_DIR, output_dir)

    cmd_det = cmd + '"voc_eval (\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d},{:d},{:s});"'.format(info['devkit_path'], 'comp4' + salt, info['image_set'], output_path, dev_kit_year, dataset_year, 'false')


    logger.info('Running:\n{}'.format(cmd_det))
    subprocess.call(cmd_det, shell=True)


def voc_info(json_dataset):
    year = json_dataset.name[4:8]
    image_set = json_dataset.name[9:]
    devkit_path = DATASETS[json_dataset.name][DEVKIT_DIR]
    assert os.path.exists(devkit_path), \
        'Devkit directory {} not found'.format(devkit_path)
    anno_path = os.path.join(
        devkit_path, 'VOC' + year, 'Annotations', '{:s}.xml')
    image_set_path = os.path.join(
        devkit_path, 'VOC' + year, 'ImageSets', 'Main', image_set + '.txt')
    return dict(
        year=year,
        image_set=image_set,
        devkit_path=devkit_path,
        anno_path=anno_path,
        image_set_path=image_set_path)
