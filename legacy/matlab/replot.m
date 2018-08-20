noid = T(strcmp('',T.Phenotype),:);
cd4 = T(strcmp('cd4',T.Phenotype),:);
cd8 = T(strcmp('cd8',T.Phenotype),:);
foxp3 = T(strcmp('foxp3',T.Phenotype),:);
macs = T(strcmp('macs',T.Phenotype),:);
other = T(strcmp('other',T.Phenotype),:);
pdl1 = T(strcmp('pd-l1',T.Phenotype),:);
pdmac = T(strcmp('pd-l1+ mac',T.Phenotype),:);
tumor = T(strcmp('tumor',T.Phenotype),:);

hold on
plot(noid.CellXPosition,-noid.CellYPosition,'.k');
plot(cd4.CellXPosition,-cd4.CellYPosition,'.m');
plot(cd8.CellXPosition,-cd8.CellYPosition,'.y');
plot(foxp3.CellXPosition,-foxp3.CellYPosition,'.g');
plot(macs.CellXPosition,-macs.CellYPosition,'.', 'color',[1 .5 0]);
plot(other.CellXPosition,-other.CellYPosition,'.w');
plot(pdl1.CellXPosition,-pdl1.CellYPosition,'.r');
plot(pdmac.CellXPosition,-pdmac.CellYPosition,'.c');
plot(tumor.CellXPosition,-tumor.CellYPosition,'.b');
set(gca,'Color',[0.4 0.4 0.4])
