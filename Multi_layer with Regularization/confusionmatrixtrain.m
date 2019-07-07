%% plotting training confusion matrix %%
confmat_train(1:10,1:10)=0;
for c=1:4000
    a=trainlabel(c);
    b=predict(c);
    
    if (a==b)
        confmat_train(a+1,a+1)=confmat_train(a+1,a+1)+1;
    else
        confmat_train(b+1,a+1)=confmat_train(b+1,a+1)+1;    
    end
end
