dirname = 'D:\ISB\HNSCC\HNSCC halle files';

files = dir(fullfile(dirname,'*.txt') );
files = {files.name}';

pattern='summary';
for i=numel(files):-1:1
    if files{i}(1)=='.'
        files(i)=[];
    end
    m=regexp(files{i},pattern,'match');
    if ~isempty(m)
        files(i)=[];
    end
end

ltable{numel(files),3}=[];
pattern1='_\d+-\d+_';
pattern2='_ES1\d \d+_';

for j=1:numel(files)
    ltable{j,1}=files{j};
    
    m=regexp(files{j},pattern1,'match');
    if isempty(m)
        m=regexp(files{j},pattern2,'match');
        ltable{j,2}=m{:}(7:end-1);
    else
        ltable{j,2}=m{:}(2:end-4);
    end
end

clinicaldata = readtable('D:\ISB\HNSCC\HNSCC halle files\clinicaldata.csv');
nrs=clinicaldata.HistoNr_;
newindex{numel(nrs),1}=[];
clpattern='\d\d\d+';
for k=1:numel(nrs)
    n=regexp(nrs{k},clpattern,'match');
    newindex{k}=n{:};
end

for i=1:length(ltable)
    loc=find(strcmp(newindex,ltable{i,2}));
    ltable{i,3} = clinicaldata{loc,2:end};
end

clearvars -except ltable