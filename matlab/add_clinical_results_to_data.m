% call script for lookup table
create_lookup_table;

% set directories
dirname = 'D:\ISB\MATLAB\output\';
dirname_scripts = 'D:\ISB\MATLAB\';

% open input file and store header
fid1 = fopen(fullfile(dirname,'allsamples_mat4.txt'),'r');
header = fgetl(fid1);

% create output file
fid2 = fopen(fullfile(dirname,'samples_result_matched.txt'),'wt');
fprintf(fid2, [header,'\tSex\tAge\tDeath\tSurvivalTime\tG\tStage\tRadiation\tChemo\tHPV\tID\n']);

% process input file
tline = fgetl(fid1);
while ischar(tline)
    %tline = strrep(tline,sprintf('\n'),''); not needed but kept in case
    C = strsplit(tline,'\t');
    loc = find(strcmp(C(1),ltable));
    
    %write to output file
    if ~isempty(ltable{loc,3})
        fprintf(fid2,'%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s\n',tline,ltable{loc,3},ltable{loc,2});
    end
    
    tline = fgetl(fid1);
end
fclose('all')

