cd4 = T(strcmp('cd4',T.Phenotype),:);
cd8 = T(strcmp('cd8',T.Phenotype),:);
foxp3 = T(strcmp('foxp3',T.Phenotype),:);
macs = T(strcmp('macs',T.Phenotype),:);
other = T(strcmp('other',T.Phenotype),:);
pdl1 = T(strcmp('pd-l1',T.Phenotype),:);
pdmac = T(strcmp('pd-l1+ mac',T.Phenotype),:);
tumor = T(strcmp('tumor',T.Phenotype),:);
    
cd4=cd4{:,{'CellXPosition','CellYPosition'}};
cd8=cd8{:,{'CellXPosition','CellYPosition'}};
foxp3=foxp3{:,{'CellXPosition','CellYPosition'}};
other=other{:,{'CellXPosition','CellYPosition'}};
macs=macs{:,{'CellXPosition','CellYPosition'}};
pdl1=pdl1{:,{'CellXPosition','CellYPosition'}};
pdmac=pdmac{:,{'CellXPosition','CellYPosition'}};
tumor=tumor{:,{'CellXPosition','CellYPosition'}};

r = 60;
figure;
hold on;

Allcells = {cd4 cd8 foxp3 other macs pdl1 pdmac tumor};
for GROUP = 1:8

    p1 = Allcells{GROUP};

    distM = zeros(length(p1),8);
    for p1cell=1:length(p1)
        for phen=1:8
            p2 = Allcells{phen};
            for p2cell=1:length(p2)
                d = sqrt((p1(p1cell,1)-p2(p2cell,1))^2 + (p1(p1cell,2)-p2(p2cell,2))^2);
                if and(d<=r,d>0)
                    distM(p1cell,phen) = distM(p1cell,phen)+1;
                end
            end
        end
    end

    subplot(2,4,GROUP), bar(sum(distM));
    Labels={'cd4','cd8','foxp3','other','macs','pdl1','pdmac','tumor'};
    %set(gca,'XTickLabel',Labels);
    %title(strcat('Distribution of cells around',{' '}, Labels{GROUP}));
    xticklabel_rotate([1:8],90,Labels);
    title(Labels{GROUP});
end