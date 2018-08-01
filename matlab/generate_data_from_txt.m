% generate list of valid txt files
dirname = 'D:\ISB\HNSCC\HNSCC halle files\';

files = dir(fullfile(dirname,'*.txt'));
files = {files.name}';

% filters out hidden files and summary files
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

% generate list of summary files
sumfiles = dir(fullfile(dirname,'*summary.txt'));
sumfiles = {sumfiles.name}';
for i=numel(sumfiles):-1:1
    if sumfiles{i}(1)=='.'
        sumfiles(i)=[];
    end
end

%% create output file
fid = fopen('D:\ISB\MATLAB\output\allsamples_mat4.txt', 'wt');
fprintf(fid,'sample\tfoxp3_near_cd8\tpdl1_near_cd8\tcd8_tumor\tcd8_stroma\tfoxp3\tfoxp3_stroma\n');

for f=1:length(files)
    %%
    %import file as table
    disp(f);
    
    %check if summary file exists, skip sample if doesn't
    summaryname = [files{f}(1:end-4), '_summary.txt'];
    if isempty(find(strcmp(summaryname,sumfiles), 1))
        continue
    end
    
    %load summmary here to extract regional area for calc cell densities
    Tsumm = readtable(fullfile(dirname,summaryname),'Delimiter','\t');
    tumorarea = Tsumm(strcmp('tumor',Tsumm.TissueCategory),:).TissueCategoryArea_pixels_;
    stromaarea = Tsumm(strcmp('stroma',Tsumm.TissueCategory),:).TissueCategoryArea_pixels_;
    
    % load sample data
    T = readtable(fullfile(dirname,files{f}),'Delimiter','\t');
    
    %subset table by cell type
    cd4 = T(strcmp('cd4',T.Phenotype),:);
    cd8 = T(strcmp('cd8',T.Phenotype),:);
    foxp3 = T(strcmp('foxp3',T.Phenotype),:);
    macs = T(strcmp('macs',T.Phenotype),:);
    other = T(strcmp('other',T.Phenotype),:);
    pdl1 = T(strcmp('pd-l1',T.Phenotype),:);
    pdmac = T(strcmp('pd-l1+ mac',T.Phenotype),:);
    tumor = T(strcmp('tumor',T.Phenotype),:);
    
    %% count cells within 30um of each cell type
    r=60;
    %create new cell array storing position of each cell
    xypos = @(x) x{:,{'CellXPosition','CellYPosition'}};
    Allcells = cellfun(xypos,{cd4 cd8 foxp3 macs other pdl1 pdmac tumor},'UniformOutput',0);
    
    circ_dist = zeros(8,8);
    for GROUP = 1:8
        p1 = Allcells{GROUP};
        distM = zeros(size(p1,1),8);
        for p1cell = 1:size(p1,1)
            for phen = 1:8
                p2 = Allcells{phen};
                for p2cell = 1:size(p2,1)
                    d = sqrt((p1(p1cell,1)-p2(p2cell,1))^2 + (p1(p1cell,2)-p2(p2cell,2))^2);
                    if and(d<=r,d>0)
                        distM(p1cell,phen) = distM(p1cell,phen)+1;
                    end
                end
            end
        end
        circ_dist(GROUP,:) = sum(distM);
    end
    %% ouput variables
    cd8foxp3 = circ_dist(2,3);
    cd8pdl1 = circ_dist(2,6)+circ_dist(2,7);
    SI = cd8foxp3 + cd8pdl1;
    
    a1 = height(cd8(strcmp('tumor',cd8.TissueCategory),:)) / tumorarea;
    a2 = height(cd8(strcmp('stroma',cd8.TissueCategory),:)) / stromaarea;
    a3 = height(foxp3);
    a4 = height(foxp3(strcmp('stroma',foxp3.TissueCategory),:)) / stromaarea;
    
    fprintf(fid,'%s\t%f\t%f\t%f\t%f\t%f\t%f\n',files{f},cd8foxp3,cd8pdl1,a1,a2,a3,a4);
    
    clear T cd4 cd8 foxp3 pdl1 tumor pdmac other macs
end
fclose('all')
clear
