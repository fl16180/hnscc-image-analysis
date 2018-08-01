%{
s1 = cd4{:,'EntireCellAlexa594Total_NormalizedCounts_TotalWeighting_'};
s2 = cd4{:,'EntireCellCy3Total_NormalizedCounts_TotalWeighting_'};

s3 = cd8{:,'EntireCellAlexa594Total_NormalizedCounts_TotalWeighting_'};
s4 = cd8{:,'EntireCellCy3Total_NormalizedCounts_TotalWeighting_'};

s12 = s1./s2; s34 = s3./s4;
%}

hist(s12,50);
h = findobj(gca,'Type','patch');
set(h,'FaceColor','r','EdgeColor','w','facealpha',0.75)
hold on;
hist(s34,50);
h1 = findobj(gca,'Type','patch');
set(h1,'facealpha',0.75);