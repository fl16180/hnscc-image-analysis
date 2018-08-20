fulldata = 'D:\ISB\MATLAB\output\samples_result_matched.txt';
T = readtable(fulldata,'Delimiter','\t');

T = T(:,[1 end 2:end-1]);

censored = ~T.Death;
result = T.SurvivalTime;

T.Death = [];
T.SurvivalTime = [];
T.HPV = [];

T = [T table(censored) table(result)];


Tmeans = grpstats(T{:,2:end},T.ID);


X = Tmeans(:,2:7);
[b,logL,h,stats] = coxphfit(X,Tmeans(:,end),'censoring',Tmeans(:,end-1));