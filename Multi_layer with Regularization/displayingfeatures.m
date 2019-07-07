%--Displaying learned features-%
U = wO;

for i=1:20
    for j = 1:10
        v = reshape(U((i-1)*10+j,:),28,28);
        subplot(10,20,(i-1)*10+j)
        image(64*v)
        colormap(gray(64));
        set(gca,'xtick',[])
        set(gca,'xticklabel',[])
        set(gca,'ytick',[])
        set(gca,'yticklabel',[])
        set(gca,'dataaspectratio',[1 1 1]);
    end
end

