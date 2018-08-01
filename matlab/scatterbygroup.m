T(strcmp(T.Phenotype,''),:)=[];

CD3marker = T{:,'EntireCellAlexa594Mean_NormalizedCounts_TotalWeighting_'};
PDL1marker = T{:,'MembraneCy5Mean_NormalizedCounts_TotalWeighting_'};
labels = T{:,'Phenotype'};
%cmap='bwymgcrk';
cmap=[0 0 1;.7 .7 .7;1 1 0;1 0 1;0 1 0;0 1 1;1 0 0;1 .5 0];
%figure();
gscatter(CD3marker,PDL1marker, labels, cmap)
%set(gca,'Color',[0.9 0.9 0.9])

