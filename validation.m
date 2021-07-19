function [ acc, prob ] = validation( Xtrn,Ytrn,Xtst,Ytst,rebalance )
%VALIDATE Summary of this function goes here
%   Detailed explanation goes here
    if rebalance
        beta_ros=1; % re-balancing ratio
        [dat_ros,lab_ros] = adasyn(Xtrn,Ytrn,beta_ros,5);
        Xtrn = [Xtrn;dat_ros];
        Ytrn = [Ytrn;lab_ros];
    end

%    Knn_EkNN = 7;
    Knn_EkNN = 5;
    [gamm,alpha] = knndsinit(Xtrn,Ytrn);
    [gamm,~] = knndsfit(Xtrn,Ytrn,Knn_EkNN,gamm,1);
    [mass,pred] = knndsval(Xtrn,Ytrn,Knn_EkNN,gamm,alpha,0,Xtst);
    [~,label] = max(mass(:,1:2),[],2);
    acc = 100*sum(label==Ytst)/length(Ytst);
    prob(:,1) = mass(:,1)+mass(:,3)./2;
    prob(:,2) = mass(:,2)+mass(:,3)./2;
end

