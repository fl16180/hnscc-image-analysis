cosD=pdist(cells,'euclidean');
TreeCos=linkage(cosD,'average');
cophenet(TreeCos,cosD);
[h,nodes]=dendrogram(TreeCos,6);

nclus=12;
lnsymb = {'b-','r-','m-','g-','c-','y-','b-','r-','m-','g-','c-','y-'};
names = {'PdL1','Cd3','Cd8','Cd163','foxP3','tumor'};
meas0 = cells ./ repmat(sqrt(sum(cells.^2,2)),1,6);
ymin = min(min(meas0));
ymax = max(max(meas0));
for i = 1:nclus
    subplot(2,6,i);
    plot(meas0(nodes==i,:)',lnsymb{i});
    %hold on;
    %plot(cmeans(i,:)','k-','LineWidth',2);
    %hold off;
    title(sprintf('Cluster %d',i));
    xlim([.9, 6.1]);
    ylim([ymin, ymax]);
    h_gca = gca;
    %set(h_gca,'XTick',1:6)
    %set(h_gca,'XTickLabel',names)
    xticklabel_rotate([1:6],90,names)
end
