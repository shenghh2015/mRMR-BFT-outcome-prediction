function [lrol, idx_fs,label,mass] = iEFS(xtrn,ytrn,xtst,B,delta,lamda,K,pf_idx)
%%
%##########################################################################
% this function robustly selects features from imbalanced training dataset,
% then predicts the outcome of testing samples via the EK-NN classifier.

% ****INPUT****:
% xtrn -- matrix (ntrn,inD) of the training instances. 
% ytrn -- vector (ntrn,1) of the corresponding class labels.
% xtst -- matrix (ntst,inD) of the testing instances.
% delta -- parameter of the penalty that controls imprecision and conflict
%          in selected feature subspace.
% lambda -- parameter of the saprisity penalty.
% B -- number of iterations for the data-rebalancing procedure.
% K -- number of nearest neighbors used in iEFS.
% pf_idx -- index of the predefined feature which should be selected.

% ****OUTPUT****:
% idx_fs -- index of selected features.
% label -- outcomes of testing samples that determined by the EK-NN in the 
%          selected feature subspace.
% mass -- matrix (ntst,c+1) describe the mass functions of testing samples.


% References:
% 
% C. Lian, S. Ruan, T. Denoeux, H. Li, and P. Vera. Robust Cancer Treatment
% Outcome Prediction Dealing with Small-Sized and Imbalanced Data from 
% FDG-PET Images, MICCAI 2016, Part II, LNCS 9901, pages 61-69, 2016.


% Date: 03 December 2016 
% Contact: Chunfeng LIAN (chunfeng.lian@{utc.fr,gmail.com})
%##########################################################################

GRP = unique(ytrn);
NGRP = length(GRP);
if NGRP~=2
    error('it should be two classes')
end

NinGRP = zeros(1,NGRP);
for i = 1 : NGRP
    NinGRP(i) = length(find(ytrn==GRP(i)));
end

w=zeros(B,size(xtrn,2));
for i=1:B
    
    %%%%%%%%%%%%%%%%%%%%%% Data Re-Balancing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Here the ADASYN method has been used, which can be replaced by other 
    % re-balancing algorithms.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    beta_ros=1; % re-balancing ratio
%     [dat_ros,lab_ros] = adasyn(xtrn,ytrn,beta_ros,3);
    [dat_ros,lab_ros] = adasyn(xtrn,ytrn,beta_ros,5);
    dat_ros = []; lab_ros = [];
    x_EFS = [xtrn;dat_ros];
    y_EFS = [ytrn;lab_ros];

    %%%%%%%%%%%%%%%%%%%%%%%% Feature Selection %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % The algorithm will be repeated iter times to avoid local minima.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    iter=1;
    fval=zeros(1,iter);
    iw=cell(1,iter);
    for r=1:iter
        [fval(r),iw{r}] = EFS(x_EFS,y_EFS,delta,lamda,K,pf_idx);
    end
    id=find(fval==min(fval));
    w(i,:)=iw{id(1)};
    
    if sum(w(i,:)==1)~=0
        idx_fs = find(w(i,:)==1);
        disp('============================================================')
        disp(['>>> features selected in this loop >>> ', num2str(idx_fs)])
        disp('============================================================')
    else
        disp('EFS failed in this loop......')
    end    
    
end

% output the most frequent feature subset
[rol,~,lrol] = unique(w,'rows');
nrol = hist(lrol,unique(lrol));
id = find(nrol==max(nrol));
feIdx = rol(id(1),:);

idx_fs = find(feIdx>=1); 
disp('*******************************************************')
disp(['->>> output feature subset ->>> ', num2str(idx_fs)])
disp('*******************************************************')

%%%%%%%%%%%%%%%%%%%% Outcome Prediction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Here the EK-NN classification rule has been used, which can also
% be replaced by other classifiers. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xidx = xtrn;
yidx = ytrn;

if sum(sum(w==1))~=0
    xidx = xtrn(:,idx_fs);
end

ck = 5;
[gamm,alpha] = knndsinit(xidx,yidx);
[gamm,~] = knndsfit(xidx,yidx,ck,gamm,1);
[mass,~] = knndsval(xidx,yidx,ck,gamm,alpha,0,xtst(:,idx_fs));
[~,label] = max(mass(:,1:2)'); label = label';
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [fval,w] = EFS(xapp,yapp,delta,lamda,K,pf_idx)

disp('running.');

V = size(xapp,2);
[xdst,ldst,lid] = feed_data(xapp,yapp);
[gamm,alpha] = knndsinit(xapp,yapp);
[gamm,~] = knndsfit(xapp,yapp,K,gamm,1);
options = gaoptimset('CrossoverFraction',0.8,'generations',200,'MigrationDirection','both', ...,
    'MigrationFraction',0.4,'FitnessScalingFcn',@fitscalingprop,'UseParallel','always', ...,
    'MigrationInterval',30,'PopulationSize',200,'StallGenLimit',30);

% options = gaoptimset('CrossoverFraction',0.8,'generations',200,'MigrationDirection','both', ...,
%     'MigrationFraction',0.4,'FitnessScalingFcn',@fitscalingprop,'UseParallel','always', ...,
%     'MigrationInterval',30,'PopulationSize',200,'StallGenLimit',30,'Display','off');

% options = gaoptimset('CrossoverFraction',0.8,'generations',200,'MigrationDirection','both', ...,
%     'MigrationFraction',0.4,'FitnessScalingFcn',@fitscalingprop,'UseParallel','always', ...,
%     'MigrationInterval',30,'PopulationSize',200,'StallGenLimit',30,'PlotFcns',@gaplotbestf); 



LB=zeros(V,1); LB(pf_idx)=1;
UB=ones(V,1); 
[w,fval] = ga(@(w)loss_bin(w,lamda,delta,xdst,ldst,lid,gamm,alpha,K,xapp,yapp),V,[],[],[],[],LB,UB,[],1:V,options);
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [xdst,ldst,lid] = feed_data(Xapp,Yapp)

[n,~] = size(Xapp); c = max(Yapp);

xdst = cell(n,1); 
ldst = zeros(n-1,n);

idx = 1:n;
for i = 1:n
    dst = ((ones(n,1)*Xapp(i,:))-Xapp);
    dst(i,:) = [];
    xdst{i,1} = dst; % features's difference
    ldst(:,i) = Yapp(idx~=i); % neighbours' labels
end

t = eye(c); lid = t(:,Yapp);
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [loss] = loss_bin(w,lamda,delta,xdst,ldst,lid,gamm,alpha,K,xapp,yapp)

n = size(ldst,2); c = 2;

xij = zeros(n-1,n); 

%-------------------------------
dst=[]; lst=[]; % use only knns
%-------------------------------

for i = 1:n
    sn = xdst{i,1}*diag(w)*(xdst{i,1})';
    xij(:,i) = diag(sn);
    
    %----------------------------------
    % use only knns
    [dss,iss] = sort(xij(:,i));
    dst = [dst dss(1:K)];
    lst = [lst ldst(iss(1:K),i)];
    %-----------------------------------
end

[gamm,alpha] = knndsinit(xapp*diag(w),yapp);
% [gamm,~] = knndsfit(xapp*diag(w),yapp,K,gamm,1);

% mij_s = alpha*exp(-(gamm(ldst).^1).*xij);
%---------------------------------------
% use only knns
mij_s = alpha*exp(-(gamm(lst).^1).*dst);
%---------------------------------------

mij_w = 1-mij_s;

mTq_w = zeros(c,n);
nr=zeros(c,n);
for i = 1:c
  %-------------------------------------
  % use only knns
  t1 = ones(size(lst)); t1(lst~=i)=0;
  t2 = zeros(size(lst)); t2(lst==i)=1;
  nr(i,:) = sum(lst==i);
  %-------------------------------------

  mTq_w(i,:) = prod(mij_w.*t1+(1-t2));
end
mTq_s = 1-mTq_w;

bba = zeros(c+2,n);

bba(1,:) = mTq_s(1,:).*mTq_w(2,:); % m(\omega1)
bba(2,:) = mTq_s(2,:).*mTq_w(1,:); % m(\omega2)
bba(3,:) = prod(mTq_w); % m(\Omega)
bba(4,:) = prod(mTq_s); % m(\empty)

%--------------------
% loss function
%--------------------
loss1 = sum(sum((bba(1:2,:)-lid).^2))/(2*n); 

loss2 = sum(bba(4,:).^2)/n+sum(bba(3,:).^2)/n;

loss3 = sum(w);

loss = loss1 + delta*loss2 + lamda*loss3;
end

%%
% adasyn: The implementation of ADASYN method.
% Input: TrainingData, Nr-by-D matrix for training data
%        TrainingLabel, Nr-by-1 vector for training class label
%        beta, the balance level
%        kNN, the number of nearest neighbors are considered
% Output: AdasynData: the generated synthetic minority class data
%         AdasynID: the generated synthetic minority class label
% Date: 02/08/2015
% By Bo Tang (btang@ele.uri.edu) and Haibo He (he@ele.uri.edu)
% For any questions and/or comments for this code/paper, please feel free
% to contact Prof. Haibo He, Electrical Engineering, University of Rhode Island,
% Email: he@ele.uri.edu
% Web:  http://www.ele.uri.edu/faculty/he/
function [adasynData, adasynID] = adasyn(trainingData, trainingLabel, beta, kNN)
numClass = length(unique(trainingLabel));
if (numClass ~= 2)
    error('error in adasyn: the input trainingLabel must be two-class!');
    return
end
[maxV, maxIX] = max([length(find(trainingLabel == 1)) length(find(trainingLabel == 2))]);
[minV, minIX] = min([length(find(trainingLabel == 1)) length(find(trainingLabel == 2))]);
majorID = maxIX;
minorID = minIX;
 
[c,v] = find(trainingLabel == majorID);
training_data_major = trainingData(c, :);
num_major = length(c);  %% number of major class data

[c,v] = find(trainingLabel == minorID);
training_data_minor = trainingData(c, :);
num_minor = length(c);  %% number of minor class data
num_data = num_major + num_minor;   %% number of all training data

N = round((num_major - num_minor) * beta);   %% number of synthetic minor class data
kNN1 = 11;
ratio = zeros(num_minor, 1);
for T = 1 : num_minor
    
    dist_all = sqrt(sum((ones(num_data, 1) * training_data_minor(T, :) - trainingData).^2, 2));
    
    [dist_sort, ind_sort] = sort(dist_all);
    ind_nn = ind_sort(2 : min(length(ind_sort), kNN1 + 1)); % changed by Ziang
    
    ind_majority = find(trainingLabel(ind_nn) == majorID);
    ind_minority = find(trainingLabel(ind_nn) == minorID);
    
    ratio(T) = length(ind_majority) / kNN1;
end

ratio_scale = ratio;
if (abs (sum(ratio_scale)) < 1e-6)
    ratio_normalized = ones(length(ratio_scale) ,1) ./ length(ratio_scale);
else
    ratio_normalized = (ratio_scale ./ sum(ratio_scale));
end

%% consider ratio_normalized as a prior pdf for sampling new data
pdf = ratio_normalized;
cumDist = cumsum(pdf);
Diff = cumDist * ones(1, N) - ones(length(pdf), 1) * rand(1, N);
Diff = (Diff <= 0) * 2 + Diff;
[C, I] = min(Diff);
sampleDataIndex = I';

adasynData =[];
ind_nn =[];
for T = 1 : length(sampleDataIndex)
    %% calc the distance of one minor class data (used for sampling new minor class data) to all other minor class data
    dist_all = sqrt(sum((ones(num_minor, 1) * training_data_minor(sampleDataIndex(T), :) - training_data_minor).^2, 2));
    
    [dist_sort, ind_sort] = sort(dist_all);
    ind_nn(T, :) = ind_sort(2 : min(length(ind_sort),kNN+1)); % changed by ziang
    
    random_select = randperm(min(size(ind_nn,2),kNN));
    ind_select = random_select(1);  % ceil(rand(1,round((N/length(training_id_minor)))) * k_nn);
    
    ind_smote = ind_nn(T, ind_select);
    
    temp_mat = training_data_minor(ind_smote, :);
    
    % create the data
    vec_smt = training_data_minor(sampleDataIndex(T), :) - temp_mat;
    adasynData(T, :) = training_data_minor(sampleDataIndex(T), :) - (rand) * (vec_smt);
    
end

adasynID = minorID * ones(size(adasynData,1), 1);

balancedTrainingDataAdaSYN = [trainingData; adasynData];

balancedTrainingLabelAdaSYN = [trainingLabel; adasynID];
end

