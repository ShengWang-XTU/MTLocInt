clc
clear
tic
seq = importdata("seq_holdout_82.csv");
pp = 1024;
for ipp = 1:length(pp)
    lenstd = pp(ipp);
    for ind = 1:length(seq)
        disp(ind)
        data = seq{ind};
        len(ind) = length(data);
        lenrat = len(ind)/lenstd;
        [xo, yo, zo] = cgr3drna_AG_T(data);
        x = mapminmax(xo, 0, 1);
        y = mapminmax(yo, 0, 1);
        z = mapminmax(zo, 0, 1);
        if lenrat >= 1 && lenrat < 4
            xu = x;
            yu = y;
            zu = z;
        elseif lenrat >= 4 && lenrat < 16
            xu = []; yu = []; zu = [];
            for i = 1:length(x)
                if x(i) <= 1/2 && y(i) <= 1/2 && z(i) <= 1/2
                    xu = [xu x(i)*2];
                    yu = [yu y(i)*2];
                    zu = [zu z(i)*2];
                else
                    xu = xu;
                    yu = yu;
                    zu = zu;
                end
            end
        elseif lenrat >= 16 && lenrat < 64
            xu = []; yu = []; zu = [];
            for i = 1:length(x)
                if x(i) <= 1/4 && y(i) <= 1/4 && z(i) <= 1/4
                    xu = [xu x(i)*4];
                    yu = [yu y(i)*4];
                    zu = [zu z(i)*4];
                else
                    xu = xu;
                    yu = yu;
                    zu = zu;
                end
            end
        elseif lenrat >= 64 && lenrat < 256
            xu = []; yu = []; zu = [];
            for i = 1:length(x)
                if x(i) <= 1/8 && y(i) <= 1/8 && z(i) <= 1/8
                    xu = [xu x(i)*8];
                    yu = [yu y(i)*8];
                    zu = [zu z(i)*8];
                else
                    xu = xu;
                    yu = yu;
                    zu = zu;
                end
            end
        elseif lenrat >= 256 && lenrat < 1024
            xu = []; yu = [];  zu = [];
            for i = 1:length(x)
                if x(i) <= 1/16 && y(i) <= 1/16 && z(i) <= 1/16
                    xu = [xu x(i)*16];
                    yu = [yu y(i)*16];
                    zu = [zu z(i)*16];
                else
                    xu = xu;
                    yu = yu;
                    zu = zu;
                end
            end
        else
            xu = x;
            yu = y;
            zu = z;
        end
        
        figxy = figure;
        plot(xu, yu, 'k.')
        pbaspect([1 1 1])
        set(gca, 'LooseInset', get(gca, 'TightInset'))
        set(gca, 'looseInset', [0 0 0 0]);
        set(gcf, 'color', 'w')
        axis off
        frame = getframe(figxy);
        img = frame2im(frame);
        eval(['imwrite(img, "CGRxy_3D_RNA_82_AG_T\CGRxy_3D_82_', num2str(ind), '.png");']);
        close all

%         figxz = figure;
%         plot(xu, zu, 'k.')
%         pbaspect([1 1 1])
%         set(gca, 'LooseInset', get(gca, 'TightInset'))
%         set(gca, 'looseInset', [0 0 0 0]);
%         set(gcf, 'color', 'w')
%         axis off
%         frame = getframe(figxz);
%         img = frame2im(frame);
%         eval(['imwrite(img, "CGRxz_3D_RNA_82_AG_T\CGRxz_3D_82_', num2str(ind), '.png");']);
%         close all
%     
%         figyz = figure;
%         plot(yu, zu, 'k.')
%         pbaspect([1 1 1])
%         set(gca, 'LooseInset', get(gca, 'TightInset'))
%         set(gca, 'looseInset', [0 0 0 0]);
%         set(gcf, 'color', 'w')
%         axis off
%         frame = getframe(figyz);
%         img = frame2im(frame);
%         eval(['imwrite(img, "CGRyz_3D_RNA_82_AG_T\CGRyz_3D_82_', num2str(ind), '.png");']);
%         close all
    end
end
toc