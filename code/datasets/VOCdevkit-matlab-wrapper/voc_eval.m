function res = voc_eval(path, comp_id, test_set, output_dir, test_kit_year, dataset_year, classy)

  VOCopts = get_voc_opts(path, test_kit_year, dataset_year, dataset_year);
  VOCopts.testset = test_set;


  year = VOCopts.dataset(4:end);

  fprintf('Testing on year %s and evaluating with kit %s \n', year, num2str(test_kit_year));
  fprintf('Data dir is %s\n', VOCopts.datadir);

  for i = 1:length(VOCopts.classes)
    cls = VOCopts.classes{i};
    res(i) = voc_evaluation(cls, VOCopts, comp_id, output_dir, test_kit_year, classy);
  end

  fprintf('\n~~~~~~~~~~~~~~~~~~~~\n');
  fprintf('Final mAP Results:\n');
  aps = [res(:).ap]';
  fprintf('%.1f, ', aps * 100);
  fprintf('%.1f', mean(aps) * 100);
  fprintf('\n~~~~~~~~~~~~~~~~~~~~\n');



function res = voc_evaluation(cls, VOCopts, comp_id, output_dir, test_kit_year, classy)

  test_set = VOCopts.testset;
  year = VOCopts.dataset(4:end);



  addpath(fullfile(VOCopts.datadir, ['VOCcode' num2str(test_kit_year)]));

  res_fn = sprintf(VOCopts.detrespath, comp_id, cls);

  recall = [];
  prec = [];
  ap = 0;
  ap_auc = 0;

  tic;

  if classy
    out_task = 'cls_';
    [recall, prec, ap] = VOCevalcls(VOCopts, comp_id, cls, false);
  else
    out_task = 'det_';
    [recall, prec, ap] = VOCevaldet(VOCopts, comp_id, cls, false);
  end;
  ap_auc = xVOCap(recall, prec);

  fprintf('!!! %s : %.4f %.4f\n', cls, ap, ap_auc);

  res.recall = recall;
  res.prec = prec;
  res.ap = ap;
  res.ap_auc = ap_auc;

  save([output_dir '/' out_task cls '_pr.mat'], 'res', 'recall', 'prec', 'ap', 'ap_auc');

  rmpath(fullfile(VOCopts.datadir, ['VOCcode' num2str(test_kit_year)]));
