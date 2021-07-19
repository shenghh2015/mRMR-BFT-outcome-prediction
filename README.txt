%1. filter-based feature selection
addpath('./FEAST/matlab')
Num_features = 13 % number of selected features
Xtrn: m x n, m training samples of n features
Ytrn: m x 1, m training samples of labels (0, 1)
featIndxSet = feast('mrmr',Num_Feats,X,Y)
%2. BFT-based feature selection
Ybft = Y+1
Xbft = Xtrn(:, featIndxSet)
delta = 1;  lambda = 0.1;  K = 5 ;  B = 5;  pf_idx=[];  % bft 
[idx_fs_list, idx_fs,~,~] = bfs(Xbft,Ybft,Xbft,B,delta,lambda,K,pf_idx)
% testing
Xtst: t x n
Ytst: t x 1
Xbft = X(:, featIndxSet)
Xfinal = Xbft(:, idx_fs)
[accREFS_trn, pREFS_trn] = validation(Xfstrn,Ytrn,Xfstrn,Ytrn,true)