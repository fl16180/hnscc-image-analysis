

%test unsupervised clustering on cells of a sample
T(strcmp(T.Phenotype,''),:)=[];


%setup
PDL1m = T{:,'MembraneCy5Mean_NormalizedCounts_TotalWeighting_'};
CD3m = T{:,'EntireCellAlexa594Mean_NormalizedCounts_TotalWeighting_'};
CD8m = T{:,'EntireCellCy3Mean_NormalizedCounts_TotalWeighting_'};
CD163m = T{:,'MembraneAlexa514Mean_NormalizedCounts_TotalWeighting_'};
Foxp3m = T{:,'NucleusFITCMean_NormalizedCounts_TotalWeighting_'};
Tumorm = T{:,'EntireCellCoumarinMean_NormalizedCounts_TotalWeighting_'};
%DAPIm = T{:,'NucleusDAPIMean_NormalizedCounts_TotalWeighting_'};

cells1=[PDL1m,CD3m,CD8m,CD163m,Foxp3m,Tumorm];
%cells=log(cells1+1);
cells=cells1;

nclus=6;
[cids, cmeans] = kmeans(cells,nclus,'dist','cosine','replicates',10,'display','final');
%[sil,h]=silhouette(cells,cids,'cosine');

%cosD=pdist(cells,'euclidean');
%TreeCos=linkage(cosD,'average');
%cophenet(TreeCos,cosD);
%[h,nodes]=dendrogram(TreeCos,0);

%{
ptsymb = {'bs','r^','md','go','c+','y*'};
for i = 1:6
    clust = find(cids==i);
    plot3(cells(clust,1),cells(clust,3),cells(clust,5),ptsymb{i});
    hold on
end
grid on
%}

lnsymb = {'b-','r-','m-','g-','c-','y-','b-','r-','m-'};
names = {'PdL1','Cd3','Cd8','Cd163','foxP3','tumor'};
%meas0 = cells ./ repmat(sqrt(sum(cells.^2,2)),1,6);

ymin = min(min(cells));
ymax = max(max(cells)-15);
for i = 1:nclus
    subplot(1,nclus,i);
    plot(cells(cids==i,:)',lnsymb{i});
    hold on;
    %plot(cmeans(i,:)','k-','LineWidth',2);
    hold off;
    title(sprintf('Cluster %d',i));
    xlim([.9, 6.1]);
    ylim([ymin, ymax]);
    h_gca = gca;
    %set(h_gca,'XTick',1:6)
    %set(h_gca,'XTickLabel',names)
    xticklabel_rotate([1:6],90,names)
end
