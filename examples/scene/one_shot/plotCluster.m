%% plot cluster
function plotCluster(res, refLabels, plotLabels)
c=99/255;
cstring=[1 0 0; 1 1 0; 0 1 0; 0 1 1; 0 0 1; 1 0 1; c 0 0; c c 0; 0 c 0; 0 0 0];
figure;
hold on;
grid on;
for ii=1:length(plotLabels);
    lb=find(refLabels==plotLabels(ii));
    plot(res(1,lb),res(2,lb),'.','Color',cstring(plotLabels(ii)+1,:));
end
legend(strsplit(num2str(plotLabels)));