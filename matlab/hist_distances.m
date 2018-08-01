function []=hist_distances(T)
    [cd4,cd8,foxp3,macs,other,pdl1,pdmac,tumor]=phenotype_setup(T);

    figure;
    hold on;
    
    Allcells = {cd4 cd8 foxp3 other macs pdl1 pdmac tumor};
    Labels={'cd4','cd8','foxp3','other','macs','pdl1','pdmac','tumor'};

    makeplots(Allcells,Labels);

end

function [cd4,cd8,foxp3,macs,other,pdl1,pdmac,tumor]=phenotype_setup(T)
    cd4 = T(strcmp('cd4',T.Phenotype),:);
    cd8 = T(strcmp('cd8',T.Phenotype),:);
    foxp3 = T(strcmp('foxp3',T.Phenotype),:);
    macs = T(strcmp('macs',T.Phenotype),:);
    other = T(strcmp('other',T.Phenotype),:);
    pdl1 = T(strcmp('pd-l1',T.Phenotype),:);
    pdmac = T(strcmp('pd-l1+ mac',T.Phenotype),:);
    tumor = T(strcmp('tumor',T.Phenotype),:);
end

function distances = calc_distances(a,b)
    type1=a{:,{'CellXPosition','CellYPosition'}};
    type2=b{:,{'CellXPosition','CellYPosition'}};

    distances = zeros(length(type1),length(type2));
    for i=1:length(type1)
        for j=1:length(type2)
            d=(type1(i,1)-type2(j,1))^2+(type1(i,2)-type2(j,2))^2;
            distances(i,j) = sqrt(d);
        end
    end

    %figure();
    %hist(distances(:),100);
end

function [] = makeplots(Allcells,Labels)
    for c1type=1:8
        for c2type=1:8
            distances = calc_distances(Allcells{c1type},Allcells{c2type});
            subplot(8,8,((c1type-1)*8+c2type)), hist(distances(:),50);
            title([Labels{c1type} ' and ' Labels{c2type}],'FontSize',10);
            set(gca,'FontSize',6)
        end
    end    
  
    %set(gca,'XTickLabel',Labels);
    %title(strcat('Distribution of cells around',{' '}, Labels{GROUP}));
    %xticklabel_rotate([1:8],90,Labels);
    
end
