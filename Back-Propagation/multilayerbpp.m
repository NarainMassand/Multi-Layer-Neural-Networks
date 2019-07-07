%-Multilayer Backpropagation by Narain Massand-%
close all;
CRtrain=[];
H=[];
load('train.mat');
load('test.mat');
load('trainlabel.mat');
load('testlabel.mat');
load('label.mat');
alpha = 0.5;
eta = 0.1;
epochs = 200;
predict = zeros(1000,1);
I = 784;

layer = ('Enter the number of Hidden layers: ');
layer =  input(layer); 
  
  for hidden=1:layer
  Y=['Hiddenlayer#',num2str(hidden)];
  disp(Y);
  a = ('Enter the number of neurons: ');
  a =  input(a);
  H(end+1)=a;
  end
H(end+1)=10; %Output layer neurons

b=zeros(layer+1);
weight={};

for i=1:layer+1
  if (i==1)
    wH=randn(I,H(i));
    wH= wH./sqrt(H(i));
  else
    wH=randn(H(i-1),H(i));  
    wH = wH./sqrt(H(i));
  end
  weight{i}=wH;
end

E = zeros(1,epochs);

for i=1:layer+1
  if (i==1)
    wM=zeros(I,H(i));
  else
    wM=zeros(H(i-1),H(i));  
  end
  mom{i}=wM;
end

tic;
disp('Training the Neural Network...');

for t = 1:epochs   
    X=['Epoch#',num2str(t)];
    disp(X);
    for k = 1:500
%             Random row selection
            r = round(length(train).*rand(1,1));
            if r == 0
                r = r+1;
            end

            % FORWARD PASS
           
            for i=1:layer+1 
                if (i==1)
                v{i} = (train(r,:)*weight{i})+b(i);
                y{i} = sigmoid(v{i});
                
                else
                v{i} = (y{i-1}*weight{i})+b(i);
                y{i} = sigmoid(v{i});        
                end
             end
            % ERROR CALCULATION
            e = ((label(r,:)-y{layer+1}));

            %BACKPROPAGATING
            for i=layer+1:-1:1
                
                if (i==1)
                sumup{i}=(weight{i+1}*change{i+1}')';
                change{i}=(sigmoid(v{i}).*(1-sigmoid(v{i}))).*sumup{i};
                delta_w{i}=eta.*train(r,:)'*change{i}+alpha.*mom{i};
                delta_b{i}=eta.*change{i};
                weight{i}=weight{i}+delta_w{i};
                b(i)=b(i)+(sum(delta_b{i})/H(i));
                elseif (i==layer+1)
                change{i}=(label(r,:)-y{i}).*(sigmoid(v{i}).*(1-sigmoid(v{i})));
                delta_w{i}=eta.*y{i-1}'*change{i}+alpha.*mom{i};
                delta_b{i}=eta.*change{i};
                weight{i}=weight{i}+delta_w{i};
                b(i)=b(i)+(sum(delta_b{i})/H(i));
                else
                sumup{i}=(weight{i+1}*change{i+1}')';
                change{i}=(sigmoid(v{i}).*(1-sigmoid(v{i}))).*sumup{i};
                delta_w{i}=eta.*y{i-1}'*change{i}+alpha.*mom{i};
                delta_b{i}=eta.*change{i};
                weight{i}=weight{i}+delta_w{i};
                b(i)=b(i)+(sum(delta_b{i})/H(i));
   
                end
            end
       end

for i=1:layer+1
    if (i==1)
    vTr = train*weight{i};
    yTr = sigmoid(vTr);
    else
    vTr = yTr*weight{i};
    yTr = sigmoid(vTr);    
    end
end
        for z = 1:length(train)
        [val, col] = max(yTr(z,:));
        predict(z,:) = col-1;
    end
%     Calculating the Classification Rate in percentage
    CRtrain(t) =(sum(predict==trainlabel)/length(train))*100;
    E(t)=100-CRtrain(t);
end
toc;
figure;
plot(1:t,CRtrain,'b--o');
title('Classification Curve');
xlabel('Epochs');
ylabel('Hitrate');
figure;
plot(1:t,E,'b--o')
title('Error Curve');
xlabel('Epochs');
ylabel('Errorrate ');