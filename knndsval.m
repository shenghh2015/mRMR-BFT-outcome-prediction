function [m,L] = knndsval(xapp,Sapp,K,gamm,alpha,loo,xtst);

% Copyright Thierry Denoeux 
% February 14, 2001
%
% KNNDSVAL: m = knndsval(xapp,Sapp,xtst,K,gamm,alpha)
%
%	K-nearest neighbour classification rule based on 
%	Dempster-Shafer theory 
% 	The decision rule corresponds together to the minimum of the
%	upper, lower and pignistic risks in the case of {0,1} costs:
%	the chosen class is the one for which the value 
%	of the Basic Probability Assignment is maximum.
%
% 	Inputs:
%
%	K : number of neighbours
% 	xapp : matrix (napp,d) of the training set 
% 	Sapp : vector (napp,1) of corresponding labels
%	gamm : vector (M,1) of parameters of the BPA
%	alpha: parameter of the BPA
%  loo: 0=validation set, 1=leave-one-out
% 	xtst : matrix (N,d) of the test set (relevant only when loo=0)
%         If loo=1, xtst is set automatically to xapp
%
% 	Outputs:
%		
% 	m: matrix (N,M+1) of final Basic Probability Assignment 
%	(M is the total number of classes, and N is the number of test 
%   examples). The first M columns contain the masses given to the 
%   singletons; the last column contains m(Omega))
%  L: vector (N,1) of predicted labels

%
% 	
%	The method can be divided in 2 steps:
%
%	* for every x of the evaluation set, computing the K-nearest 
%	neighbours among the training set vectors.
%	* computing the BPA of the evaluation vectors
%
%	See also: KNNDSINIT,KNNDSFIT
%
% References:
% 
% T. Denoeux. A k-nearest neighbor classification rule based on 
%  Dempster-Shafer theory. IEEE Transactions on Systems, Man
%  and Cybernetics, 25(05):804-813, 1995.
%
% L. M. Zouhal and T. Denoeux. An evidence-theoretic k-NN rule with 
% parameter optimization. IEEE Transactions on Systems, Man and 
% Cybernetics - Part C, 28(2):263-271,1998.




[Napp,nent]=size(xapp);
M=max(Sapp);

if loo,
   xtst=xapp;
end;

[Ntst,nent]=size(xtst);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computation of the K-nearest neighbours in the training set

  dst=[];ist=[];
  for i = 1:Ntst,
      if nent > 1
          dist=sum(((ones(Napp,1)*xtst(i,:))-xapp)'.^2)';
      else dist=(abs(ones(Napp,1)*(xtst(i,:)) - xapp)).^2;
      end
      [dss,iss]=sort(dist);
      dst = [dst dss(1+loo:K+loo)];
      ist = [ist iss(1+loo:K+loo)];
  end;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computation of the BPA

m = classdstst(alpha,gamm,xtst,dst,ist,Sapp,K); 
[temp,L]=max(m(:,1:M)');
L=L';



%%%%%%%%%%%% dependent programs%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function m = classdstst(alpha,gamm,xtst,ds,is,Sapp,K);

% CLASSDSTST:  m = classds(alpha,gamm,ds,is,Sapp,K)
%
%	This program provides the BPA, the labels and the
%	error rate  in a K-nearest neighbour rule based on 
%	Dempster-Shafer theory.  
% 		
% 	alpha (1,1), gamm (M,1): parameters of the BPA
%	(M is the number of classes)
% 	ds, is : matrices (K,N) containing respectively 
%	distances et indices of the K - nearest neighbours 
%	of N vectors belonging to the test set. 
%	Sapp: vector (napp,1) of labels of the training set.
%	K: number of neighbours
%
% 	m: matrix (M+1,N) of the final BPA 

N= size(xtst,1);
M=max(Sapp);
m = [zeros(M,N);ones(1,N)]; 
cppv=zeros(N,1);
	
for i=1:N,
   for j=1:K,
     m1 = zeros(M+1,1);
     m1(Sapp(is(j,i))) = alpha*exp(-gamm(Sapp(is(j,i))).^2*ds(j,i));
     m1(M+1) = 1 - m1(Sapp(is(j,i)));
     m(1:M,i) = m1(1:M).*m(1:M,i) + m1(1:M)*m(M+1,i) + m(1:M,i)*m1(M+1);
     m(M+1,i) = m1(M+1) * m(M+1,i);    
  end;
end;
m=m./(ones(M+1,1)*sum(m));
m=m';


