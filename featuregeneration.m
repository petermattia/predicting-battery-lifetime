%% Peter Attia, Marc Deetjen, Jeremy Witmer
% cs229 project
% Windows 10 path conventions

cells = 85:124; % training data = 1:84; testing data = 85:124
filename = 'testing';
doPlot = 0;

%% Load data

close all; clc
loadData = 0;
if loadData
    path = 'D:\Data_Matlab\Batch_data';
    load([path,'\2017-05-12_batchdata_updated_struct_errorcorrect'])
    
    batch1 = batch;
    numBat1 = size(batch1,2);
    
    load([path,'\2017-06-30_batchdata_updated_struct_errorcorrect'])
    
    %remove batteries continuing from batch 1
    batch([8:10,16:17]) = [];
    batch2 = batch;
    numBat2 = size(batch2,2);
    clearvars batch
    
    load([path,'\2018-04-12_batchdata_updated_struct_errorcorrect'])
    batch8 = batch;
    batch8(38) = []; %remove channel 46 upfront
    numBat8 = size(batch8,2);
    endcap8 = zeros(numBat8,1);
    clearvars batch
    for i = 1:numBat8
        endcap8(i) = batch8(i).summary.QDischarge(end);
    end
    rind = find(endcap8 > 0.885);
    batch8(rind) = [];
    
    %remove the noisy Batch 8 batteries
    nind = [3, 40:41];
    batch8(nind) = [];
    numBat8 = size(batch8,2);
    
    batch = [batch1, batch2, batch8];
    numBat = numBat1 + numBat2 + numBat8;
    
    %remove the batteries that do not finish in Batch 1
    batch([9,11,13,14,23]) = [];
    numBat = numBat - 5;
    numBat1 = numBat1 - 5;
    
    clearvars -except batch numBat1 numBat2 numBat8 numBat
end

%extract the number of cycles to 0.88
cycle_lives = zeros(numBat,1);
for i = 1:numBat
    if batch(i).summary.QDischarge(end) < 0.88
        cycle_lives(i) = find(batch(i).summary.QDischarge < 0.88,1);
        
    else
        cycle_lives(i) = size(batch(i).cycles,2) + 1;
    end
    
end

cycle_lives(1:5) = [1852; 2160; 2237; 1434; 1709];

%% Start Marc's code
yPlt = [1,5]; % Which number of cycle index to plot
cPlt = [50,55,60]; % Which cells to plot

% Search domain
LastCycle = 20:10:100; % Start at 2 and go till this number

% Variables
nCy = length(LastCycle); % Number of different cycle lengths
nV = length(batch(1).cycles(2).Qdlin); % Number of voltage sample points
nC = length(cells); % Number of cells we are looking at

cycle_lives = cycle_lives(cells);

%% Generate heatmaps (raw & normalized)
% Key variables
cycles = cell(1,nCy); % List of cycles for each iteration
Qheatmap_raw = cell(nC,nCy);
Qheatmap_NormEach = cell(nC,nCy); % Normalized by each individual cell (median)
Qheatmap_NormTot = cell(nC,nCy); % Normalized by the average over all cells (mean)

% Other variables
Qheatmap_tot = cell(1,nCy); % Normalization for the average over all cells
for y = 1:nCy % Iterate over different number of cycle inclusions
    cycles{y} = 2:LastCycle(y);
    Qheatmap_tot{y} = zeros(nV,length(cycles{y}));
    for c = 1:nC
        k = cells(c);
        Qheatmap_raw{c,y} = zeros(nV,length(cycles{y}));
        for i = 1:length(cycles{y})
            i2 = cycles{y}(i);
            Qheatmap_raw{c,y}(:,i) = batch(k).cycles(i2).Qdlin;
        end
        Qheatmap_NormEach{c,y} = Qheatmap_raw{c,y} - median(Qheatmap_raw{c,y},2)*ones(1,length(cycles{y}));
        Qheatmap_tot{y} = Qheatmap_tot{y} + Qheatmap_raw{c,y};
    end
    Qheatmap_tot{y} = Qheatmap_tot{y} / nC;
    for c = 1:nC
        Qheatmap_NormTot{c,y} = Qheatmap_raw{c,y} - Qheatmap_tot{y};
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Figures %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
if  doPlot
    for y = yPlt
        figure; imagesc(Qheatmap_tot{y},[-0.03 0.03]); colorbar; title('Batch heat map')
        for x = cPlt
            c = find(cPlt==x);
            figure('units','normalized','outerposition',[0 0 1 1]);
            subplot(2,2,1); imagesc(Qheatmap_NormEach{c,y},[-0.03 0.03]); colorbar; title(['Cycle life = ' num2str(batch(k).cycle_life)])
            subplot(2,2,2); plot(Qheatmap_NormEach{c,y}(400,:))
            subplot(2,2,3); imagesc(Qheatmap_NormTot{c,y},[-0.03 0.03]); colorbar
            subplot(2,2,4); plot(Qheatmap_NormTot{c,y}(400,:))
        end
    end
end

%% Noise elimination
% Key parameters to set
NE_InlierDist = 0.007; % Set higher to filter out less noise
NE_LimV = [230,nV]; % Voltage limits [Min,Max]
NE_LimY = [8,0]; % cYcle limits [Min,Max] (set second parameter to 0 if it should go till the end)

% Key variables
Qheatmap_NormNE = cell(nC,nCy); % Outliers replaced with surface fit with 15 paramters

% Other variables
FncFit44 = cell(nC,nCy); % RANSAC surface fit with 15 parameters
for y = 1:nCy
    fprintf('\nNoiseElim: y = %d / %d',y,nCy);
    NE_LimY0 = NE_LimY;
    if NE_LimY(2)==0, NE_LimY0(2)=length(cycles{y}); end
    for c = 1:nC
        fprintf('.');
        % Set up data subject to limits set above for Voltage and cYcles
        Qadj = Qheatmap_NormEach{c,y};
        V = repmat(1:nV,length(cycles{y}),1)';
        Y = repmat(cycles{y},nV,1);
        Qadj = Qadj(NE_LimV(1):NE_LimV(2),NE_LimY0(1):NE_LimY0(2));
        V = V(NE_LimV(1):NE_LimV(2),NE_LimY0(1):NE_LimY0(2));
        Y = Y(NE_LimV(1):NE_LimV(2),NE_LimY0(1):NE_LimY0(2));

        % Complex model for oulier filling
        Pts = [V(:),Y(:),Qadj(:)];
        FitType = 'poly22'; % 1+2+3+4+5=15 parameters
        IterNum = 30;
        [FncFit44{c,y},In44] = ransacFncAny(Pts,FitType,IterNum,NE_InlierDist);
        
        % Replace outliers with FncFit44
        Out44 = setdiff(1:size(Pts,1),In44); % Outliers
        NE_Pts = Pts;
        NE_Pts(Out44,3) = FncFit44{c,y}(Pts(Out44,1:2));
        NE_Q_Lim = reshape(NE_Pts(:,3),size(V));
        Qheatmap_NormNE{c,y} = Qheatmap_NormEach{c,y};
        Qheatmap_NormNE{c,y}(NE_LimV(1):NE_LimV(2),NE_LimY0(1):NE_LimY0(2)) = NE_Q_Lim;
        
        % Figures: simple surface fit
        if  doPlot% && ismember(y,yPlt) && ismember(c,cPlt)
            figure('units','normalized','outerposition',[0 0 1 1]);
            subplot(1,2,1); title('Complex RANSAC surface fit')
            surf(V,Y, Qadj, 'FaceAlpha',0.5); hold on; shading interp; set(gca,'clipping','off');
            plot3(Pts(In44,1),Pts(In44,2),Pts(In44,3),'k.','MarkerSize',0.1); hold on;
            s2 = surf(V,Y, reshape(FncFit44{c,y}(Pts(:,1:2)),size(V)), 'FaceAlpha',0.5); hold on; shading interp; set(gca,'clipping','off'); set(s2,'FaceColor',[1,0,0]);
            AveEr = sum((FncFit44{c,y}(Pts(In44,1:2)) - Pts(In44,3)).^2)^0.5; title(['polyN2, AveEr=',num2str(AveEr)]);
            subplot(1,2,2); title('Noise Elimination')
            surf(V,Y, NE_Q_Lim, 'FaceAlpha',0.5); hold on; shading interp; set(gca,'clipping','off');
            plot3(V(:),Y(:),Qadj(:),'k.','MarkerSize',0.1); hold on;
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Figures %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if  doPlot
    for y = yPlt
        for x = cPlt
            c = find(cPlt==x);
            figure('units','normalized','outerposition',[0 0 1 1]);
            subplot(2,2,1); imagesc(Qheatmap_NormEach{c,y},[-0.03 0.03]); colorbar; title(['Cycle life = ' num2str(batch(k).cycle_life)])
            subplot(2,2,2); plot(Qheatmap_NormEach{c,y}(400,:))
            subplot(2,2,3); imagesc(Qheatmap_NormNE{c,y},[-0.03 0.03]); colorbar; title(['Cycle life = ' num2str(batch(k).cycle_life)])
            subplot(2,2,4); plot(Qheatmap_NormNE{c,y}(400,:))
        end
    end
end

%% Features: Simple surface fit
% Key parameters to set
InlierDist = 0.003;
LimV = [230,nV]; % Voltage limits [Min,Max]
LimY = [8,0]; % cYcle limits [Min,Max] (set second parameter to 0 if it should go till the end)

% Key variables
F_SurfFit3 = zeros(nC,nCy,3); % RANSAC: Q = p00 + p01*y + p11*x*y where x = V(voltage), y = cYcle

% Other variables
FncFit3 = cell(nC,nCy); % RANSAC surface fit with 3 parameters
for y = 1:nCy
    fprintf('\nSurfFeatures: y = %d / %d',y,nCy);
    LimY0 = LimY;
    if LimY(2)==0, LimY0(2)=length(cycles{y}); end
    for c = 1:nC
        fprintf('.');
        % Set up data subject to limits set above for Voltage and cYcles
        Qadj = Qheatmap_NormEach{c,y};
        V = repmat(1:nV,length(cycles{y}),1)';
        Y = repmat(cycles{y},nV,1);
        Qadj = Qadj(LimV(1):LimV(2),LimY0(1):LimY0(2));
        V = V(LimV(1):LimV(2),LimY0(1):LimY0(2));
        Y = Y(LimV(1):LimV(2),LimY0(1):LimY0(2));

        % Best simple model
        Pts = [V(:),Y(:),Qadj(:)];
        FitType = fittype('p00 + p01*y + p11*x*y', 'dependent',{'z'},'independent',{'x','y'});
        IterNum = 30;
        warning('off','curvefit:fit:noStartPoint'); % Turn off warning about no starting point... maybe fix this later and add a starting point for surface fit
        [FncFit3{c,y},In3] = ransacFncAny(Pts,FitType,IterNum,InlierDist);
        F_SurfFit3(c,y,:) = [FncFit3{c,y}.p00, FncFit3{c,y}.p01, FncFit3{c,y}.p11]; 
        
        % Figures: simple surface fit
        if  doPlot %&& ismember(y,yPlt) && ismember(c,cPlt)
            figure;
            surf(V,Y, Qadj, 'FaceAlpha',0.5); hold on; shading interp; set(gca,'clipping','off');
            plot3(Pts(In3,1),Pts(In3,2),Pts(In3,3),'k.','MarkerSize',0.1); hold on;
            s2 = surf(V,Y, reshape(FncFit3{c,y}(Pts(:,1:2)),size(V)), 'FaceAlpha',0.5); hold on; shading interp; set(gca,'clipping','off'); set(s2,'FaceColor',[1,0,0]);
            AveEr = sum((FncFit3{c,y}(Pts(In3,1:2)) - Pts(In3,3)).^2)^0.5; title(['polyN2, AveEr=',num2str(AveEr)]);
        end
    end
end

%% Features: other
slope_2pt9 = zeros(nC, nCy);
intercept_2pt9 = zeros(nC, nCy);
 
slope_2pt9_corr = zeros(nC, nCy);
intercept_2pt9_corr = zeros(nC, nCy);
 
ave_vert_slice_mean = zeros(nC, nCy);
ave_vert_slice_var = zeros(nC, nCy);
ave_vert_slice_skewness = zeros(nC, nCy);
ave_vert_slice_kurtosis = zeros(nC, nCy);
 
heatmap_mean_abs = zeros(nC, nCy);
heatmap_mean = zeros(nC, nCy);
heatmap_var = zeros(nC, nCy);

heatmap_mean_abs_corr = zeros(nC, nCy);
heatmap_mean_corr = zeros(nC, nCy);
heatmap_var_corr = zeros(nC, nCy);

DeltaQ_min = zeros(nC, nCy);
DeltaQ_mean = zeros(nC, nCy);
DeltaQ_var = zeros(nC, nCy);
DeltaQ_skew = zeros(nC, nCy);
DeltaQ_kurt = zeros(nC, nCy);
DeltaQ_2V = zeros(nC, nCy);

% Original features
init_cap = zeros(nC, nCy);
max_minus_2 = zeros(nC, nCy);
final_cap = zeros(nC, nCy);
Tmax = zeros(nC, nCy);
Tmin = zeros(nC, nCy);
chargetime = zeros(nC, nCy);
linfit_all_int = zeros(nC, nCy);
linfit_all_slope = zeros(nC, nCy);
linfit_last10_int = zeros(nC, nCy);
linfit_last10_slope = zeros(nC, nCy);
minIR = zeros(nC, nCy);
IR2 = zeros(nC, nCy);
IRdiff = zeros(nC, nCy);

% Feature generation loop
for y = 1:nCy
	for c = 1:nC
	    k = cells(c);
	    i = 1;
	    j = length(cycles{y});
	    
	    % Slice slope
	    slice_itoj_cellbg = Qheatmap_raw{c,y}(400,i:j);
	    p1 = polyfit(1:length(slice_itoj_cellbg), slice_itoj_cellbg,1);
	    slope_2pt9(c,y) = p1(1);
	    intercept_2pt9(c,y) = p1(2);
	    
	    % Slice slope normalized and corrected
	    slice_itoj_totbg = Qheatmap_NormNE{c,y}(400,i:j);
	    slice_itoj_totbg = sum(slice_itoj_totbg,1);
	    p2 = polyfit(1:length(slice_itoj_totbg), slice_itoj_totbg,1);
	    slope_2pt9_corr(c,y) = p2(1);
	    intercept_2pt9_corr(c,y) = p2(2);
	    
	    % Heatmap
	    heatmap_mean_abs(c,y) = mean(abs(Qheatmap_raw{c,y}(:)));
	    heatmap_mean(c,y) = mean(Qheatmap_raw{c,y}(:));
	    heatmap_var(c,y) = var(Qheatmap_raw{c,y}(:));
        
	    % Heatmap
	    heatmap_mean_abs_corr(c,y) = mean(abs(Qheatmap_NormNE{c,y}(:)));
	    heatmap_mean_corr(c,y) = mean(Qheatmap_NormNE{c,y}(:));
	    heatmap_var_corr(c,y) = var(Qheatmap_NormNE{c,y}(:));
	    
	    % Vert difference (DeltaQ)
        DeltaQ = (batch(k).cycles(cycles{y}(end)).Qdlin - batch(k).cycles(10).Qdlin);
        DeltaQ_min(c,y) = min(DeltaQ);
	    DeltaQ_mean(c,y) = mean(DeltaQ);
	    DeltaQ_var(c,y) = var(DeltaQ);
	    DeltaQ_skew(c,y) = skewness(DeltaQ);
	    DeltaQ_kurt(c,y) = kurtosis(DeltaQ);
        DeltaQ_2V(c,y) = DeltaQ(1000);
        
        %% Original paper features
        init_cap(c,y) = batch(k).summary.QDischarge(2);
        max_minus_2(c,y) = max(batch(k).summary.QDischarge(1:cycles{y}(end))) - batch(k).summary.QDischarge(2);
        final_cap(c,y) = batch(k).summary.QDischarge(cycles{y}(end));
        Tmax(c,y) = max(batch(k).summary.Tmax(2:cycles{y}(end)));
        Tmin(c,y) = min(batch(k).summary.Tmin(2:cycles{y}(end)));
        chargetime(c,y) = mean(batch(k).summary.chargetime(2:6));
        
        R3 = regress(batch(k).summary.QDischarge(2:cycles{y}(end)),[2:cycles{y}(end);ones(1,length(2:cycles{y}(end)))]');
        linfit_all_int(c,y) = R3(1);
        linfit_all_slope(c,y) = R3(2);
        
        cyclerange = cycles{y}(end)-9:cycles{y}(end);
        R2 = regress(batch(k).summary.QDischarge(cyclerange),[cyclerange;ones(1,10)]');
        linfit_last10_int(c,y) = R2(1);
        linfit_last10_slope(c,y) = R2(2);
        
        % IR features
        minIR(c,y) = min(batch(k).summary.IR((batch(k).summary.IR(1:100) > 0)));
        IR2(c,y) = batch(k).summary.IR(2);
        IRdiff(c,y) = batch(k).summary.IR(100) - batch(k).summary.IR(2);
	end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Figures %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if doPlot
	for y = yPlt
		figure
		subplot(2,2,1)
		plot(slope_2pt9(:,y), cycle_lives,'o'), hold on
		slope_cell_bg = corrcoef(slope_2pt9(:,y), cycle_lives);
		title(['Slope cell bg, corr = ', num2str(slope_cell_bg(1,2))])
		 
		subplot(2,2,2)
		plot(intercept_2pt9(:,y), cycle_lives,'o')
		intercept_cell_bg = corrcoef(intercept_2pt9(:,y), cycle_lives);
		title(['Intercept cell bg, corr = ', num2str(intercept_cell_bg(1,2))])
		 
		subplot(2,2,3)
		plot(slope_2pt9_corr(:,y), cycle_lives,'o')
		slope_tot_bg = corrcoef(slope_2pt9_corr(:,y), cycle_lives);
		title(['Slope tot bg, corr = ', num2str(slope_tot_bg(1,2))])
		 
		subplot(2,2,4)
		plot(intercept_2pt9_corr(:,y), cycle_lives,'o')
		intercept_tot_bg = corrcoef(intercept_2pt9_corr(:,y), cycle_lives);
		title(['Intercept tot bg, corr = ', num2str(intercept_tot_bg(1,2))])
		 
		% Ave vert slice
		figure
		subplot(2,2,1)
		plot(ave_vert_slice_mean(:,y), cycle_lives,'o')
		ave_vert_slice_mean_cc = corrcoef(ave_vert_slice_mean(:,y), cycle_lives);
		title(['Ave vert slice mean, corr = ', num2str(ave_vert_slice_mean_cc(1,2))])
		 
		subplot(2,2,2)
		plot(ave_vert_slice_var(:,y), cycle_lives,'o')
		ave_vert_slice_var_cc = corrcoef(ave_vert_slice_var(:,y), cycle_lives);
		title(['Ave vert slice var, corr = ', num2str(ave_vert_slice_var_cc(1,2))])
		
		subplot(2,2,3)
		plot(ave_vert_slice_skewness(:,y), cycle_lives,'o')
		ave_vert_slice_skewness_cc = corrcoef(ave_vert_slice_skewness(:,y), cycle_lives);
		title(['Ave vert slice skew, corr = ', num2str(ave_vert_slice_skewness_cc(1,2))])
		 
		subplot(2,2,4)
		plot(ave_vert_slice_kurtosis(:,y), cycle_lives,'o')
		ave_vert_slice_kurt_cc = corrcoef(ave_vert_slice_kurtosis(:,y), cycle_lives);
		title(['Ave vert slice kurt, corr = ', num2str(ave_vert_slice_kurt_cc(1,2))])
		 
		 
		% Summary
		figure
		subplot(2,2,1)
		plot(heatmap_mean_abs(:,y), cycle_lives,'o')
		summary_mean_abs_cc = corrcoef(heatmap_mean_abs(:,y), cycle_lives);
		title(['Summary mean abs, corr = ', num2str(summary_mean_abs_cc(1,2))])
		 
		subplot(2,2,2)
		plot(heatmap_mean(:,y), cycle_lives,'o')
		summary_mean_cc = corrcoef(heatmap_mean(:,y), cycle_lives);
		title(['Summary mean, corr = ', num2str(summary_mean_cc(1,2))])
		 
		subplot(2,2,3)
		plot(heatmap_var(:,y), cycle_lives,'o')
		summary_var_cc = corrcoef(heatmap_var(:,y), cycle_lives);
		title(['Summary var, corr = ', num2str(summary_var_cc(1,2))])
		 
		subplot(2,2,4)
		plot(heatmap_var_corr(:,y), cycle_lives,'o')
		summary_var_bl_cc = corrcoef(heatmap_var_corr(:,y), cycle_lives);
		title(['Summary var bl, corr = ', num2str(summary_var_bl_cc(1,2))])
		 
		% Delta Q_(40 - 10)
		figure
		subplot(2,2,1)
		plot(DeltaQ_mean(:,y), cycle_lives,'o')
		DeltaQ_mean_cc = corrcoef(DeltaQ_mean(:,y), cycle_lives);
		title(['DeltaQ mean, corr = ', num2str(DeltaQ_mean_cc(1,2))])
		 
		subplot(2,2,2)
		plot(DeltaQ_var(:,y), cycle_lives,'o')
		DeltaQ_var_cc = corrcoef(DeltaQ_var(:,y), cycle_lives);
		title(['DeltaQ var, corr = ', num2str(DeltaQ_var_cc(1,2))])
		 
		subplot(2,2,3)
		plot(DeltaQ_skew(:,y), cycle_lives,'o')
		DeltaQ_skew_cc = corrcoef(DeltaQ_skew(:,y), cycle_lives);
		title(['DeltaQ skew, corr = ', num2str(DeltaQ_skew_cc(1,2))])
		 
		subplot(2,2,4)
		plot(DeltaQ_kurt(:,y), cycle_lives,'o')
		DeltaQ_kurt_cc = corrcoef(DeltaQ_kurt(:,y), cycle_lives);
		title(['DeltaQ kurt, corr = ', num2str(DeltaQ_kurt_cc(1,2))])
	end
end

%% Export features

% Variable names
VarNames = {'CellNum','cycle_lives',...
    'SurfFit_p1','SurfFit_p2','SurfFit_p3'...
    'slope_2pt9V', 'int_2pt9V',...
    'slope_2pt9V_corr', 'int_2pt9V_corr',...
    'heatmap_mean_abs', 'heatmap_mean', 'heatmap_var', ...
    'heatmap_mean_abs_corr', 'heatmap_mean_corr', 'heatmap_var_corr',...
    'DeltaQ_min','DeltaQ_mean', 'DeltaQ_var', 'DeltaQ_skew', 'DeltaQ_kurt', 'DeltaQ_2V',...
    'Q_cycle2','Q_lastcycle','Qmax_minus_Q2','chargetime','Tmax','Tmin','Qlinfit_allcycles_int','Qlinfit_allcycles_slope',...
    'Qlinfit_last10cycles_int','Qlinfit_last10cycles_slope','IR_min','IR_cycle2', 'IR_diff'};
l = length(VarNames);
VarNames_log = VarNames;
for k = 3:length(VarNames)
    VarNames_log{l+k-2} = ['log_',VarNames{k}];
end

% Initialization
Array = cell(1,nCy);
Array_log = cell(1,nCy);

for y = 1:nCy
    Array{y} = [cells', cycle_lives, ...
        reshape(F_SurfFit3(:,y,:),nC,3)...
        slope_2pt9(:,y), intercept_2pt9(:,y),...
        slope_2pt9_corr(:,y), intercept_2pt9_corr(:,y),...
        heatmap_mean_abs(:,y), heatmap_mean(:,y), heatmap_var(:,y), ...
        heatmap_mean_abs_corr(:,y), heatmap_mean_corr(:,y), heatmap_var_corr(:,y), ...
        DeltaQ_min(:,y), DeltaQ_mean(:,y), DeltaQ_var(:,y), DeltaQ_skew(:,y), DeltaQ_kurt(:,y), ...
        DeltaQ_2V(:,y),init_cap(:,y),final_cap(:,y),max_minus_2(:,y),chargetime(:,y),Tmax(:,y), Tmin(:,y),...
        linfit_all_int(:,y), linfit_all_slope(:,y), linfit_last10_int(:,y), linfit_last10_slope(:,y),...
        minIR(:,y),IR2(:,y), IRdiff(:,y)];
    Array_log{y} = [Array{y} log10(abs(Array{y}(:,3:end)))]; %add log10 features
    
    TableCSV = array2table(Array{y},'VariableNames',VarNames);
    FileName = ['./features/cycles_',num2str(cycles{y}(1)),'TO',num2str(cycles{y}(end)),'.csv'];
    writetable(TableCSV,FileName);
    
    TableCSV = array2table(Array_log{y},'VariableNames',VarNames_log);
    FileName = ['./features/cycles_',num2str(cycles{y}(1)),'TO',num2str(cycles{y}(end)),'_log.csv'];
    writetable(TableCSV,FileName);
end

zip([filename,'.zip'],'features');

%% Performance (correlation coeff)
nFeatures = size(Array{1},2)-2;
CorrCoeff = zeros(nFeatures,nCy);
CorrCoeff_log = zeros(size(Array{1},2)-2,nCy);
for y = 1:nCy
    CC = corrcoef(Array{y}(:,2:end));
    CorrCoeff(:,y) = CC(1,2:end);
    CC_log = corrcoef([log(Array{y}(:,2)),Array{y}(:,3:end)]);
    CorrCoeff_log(:,y) = CC_log(1,2:end);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Figures %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if doPlot
    F_PltBreaks = [1,5,10,15,nFeatures];
    figure('units','normalized','outerposition',[0 0 1 1]);
    for i = 1:length(F_PltBreaks)-1
        subplot(1,4,i);
        for f = F_PltBreaks(i):F_PltBreaks(i+1)
            plot(LastCycle,abs(CorrCoeff(f,:)), 'DisplayName',VarNames{f+2}); hold on;
        end
        legend('Location','SouthWest');
        set(gca,'ylim',[0,1]); title('Correlation Coeffs');
    end
    
    F_PltBreaks = [1,5,10,15,nFeatures];
    figure('units','normalized','outerposition',[0 0 1 1]);
    for i = 1:length(F_PltBreaks)-1
        subplot(1,4,i);
        for f = F_PltBreaks(i):F_PltBreaks(i+1)
            plot(LastCycle,abs(CorrCoeff_log(f,:)), 'DisplayName',VarNames{f+2}); hold on;
        end
        legend('Location','SouthWest');
        set(gca,'ylim',[0,1]); title('Log Correlation Coeffs');
    end
end

%y = nCy;
y = 3;
for f = 1:nFeatures
    fprintf('\nCorr = %f | Corr log = %f | %s',CorrCoeff(f,y),CorrCoeff_log(f,y),VarNames{f+2});
end
fprintf('\n\n');