%% plot cluster
function plotCluster3D(res, refLabels, plotLabels)
c=99/255;
cstring=[1 0 0; 1 1 0; 0 1 0; 0 1 1; 0 0 1; 1 0 1; c 0 0; c c 0; 0 c 0; 0 0 0];
figure;
grid on;
for ii=1:length(plotLabels);
    lb=find(refLabels==plotLabels(ii));
    scatter3(res(1,lb),res(2,lb),res(3,lb),'.','MarkerEdgeColor',cstring(plotLabels(ii)+1,:));
    hold on;
end
legend(strsplit(num2str(plotLabels)));