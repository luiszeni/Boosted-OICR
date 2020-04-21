function VOCopts = get_voc_opts(path, dev_kit_year, dataset_year)

tmp = pwd;
cd(path);
code_folder = [path '/VOCcode' num2str(dev_kit_year) '/'];

%fprintf('path %s  %s\n', path, code_folder)
try
  %fprintf('location %s\n',pwd)

  addpath(code_folder);
  %fprintf('whut? %s\n',code_folder)
  if dataset_year == 2007
  	VOCinit2007;
  else
  	VOCinit2012;
  end;
catch
  rmpath(code_folder);
  cd(tmp);
  error(sprintf('VOCcode directory not found under %s', path));
end
rmpath(code_folder);
cd(tmp);
