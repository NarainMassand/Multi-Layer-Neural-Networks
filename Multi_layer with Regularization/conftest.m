%% plotting testing confusion matrix %%
confmat_test(1:10,1:10)=0;
for d=1:1000
    a=testlabel(d);
    b=predicttest(d);
    
    if (a==b)
        confmat_test(a+1,a+1)=confmat_test(a+1,a+1)+1;
    else
        confmat_test(b+1,a+1)=confmat_test(b+1,a+1)+1;    
    end
end
