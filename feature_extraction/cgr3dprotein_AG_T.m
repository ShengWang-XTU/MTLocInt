% Protein 3D CGR
function [x, y, z] = cgr3dprotein_AG_T(seq20)
    % seq20 = 'MSGGGVIRGPAGNNDCRIYVGNLPPDIRTKDIEDVFYKYGAIRDIDLKNRRGGPPFAFV';
    for s = 1:length(seq20)
        if (seq20(s) == 'A')||(seq20(s) == 'V')||(seq20(s) == 'L')||(seq20(s) == 'I')
            seq4(s) = 'A';
        elseif (seq20(s) == 'P')||(seq20(s) == 'M')||(seq20(s) == 'W')||(seq20(s) == 'F')
            seq4(s) = 'A';
        elseif (seq20(s) == 'Q')||(seq20(s) == 'S')||(seq20(s) == 'T')||(seq20(s) == 'C')
            seq4(s) = 'C';
        elseif (seq20(s) == 'N')||(seq20(s) == 'Y')||(seq20(s) == 'G')
            seq4(s) = 'C';
        elseif (seq20(s) == 'D')||(seq20(s) == 'E')
            seq4(s) = 'G';
        elseif (seq20(s) == 'K')||(seq20(s) == 'R')||(seq20(s) == 'H')
            seq4(s) = 'T';
        end
    end
    A = ([0, 0, 0]-(1/2))*2*sqrt(1/3);
    T = ([1, 1, 0]-(1/2))*2*sqrt(1/3);
    G = ([1, 0, 1]-(1/2))*2*sqrt(1/3);
    C = ([0, 1, 0]-(1/2))*2*sqrt(1/3);
    if seq4(1) == 'A'
        x(1) = mean([0, A(1)]);
        y(1) = mean([0, A(2)]);
        z(1) = mean([0, A(3)]);
    elseif seq4(1) == 'T'
        x(1) = mean([0, T(1)]);
        y(1) = mean([0, T(2)]);
        z(1) = mean([0, T(3)]);
    elseif seq4(1) == 'C'
        x(1) = mean([0, C(1)]);
        y(1) = mean([0, C(2)]);
        z(1) = mean([0, C(3)]);
    elseif seq4(1) == 'G'
        x(1) = mean([0, G(1)]);
        y(1) = mean([0, G(2)]);
        z(1) = mean([0, G(3)]);
    end
    for i = 2:length(seq4)
        if seq4(i) == 'A'
            x(i) = mean([x(i-1), A(1)]);
            y(i) = mean([y(i-1), A(2)]);
            z(i) = mean([z(i-1), A(3)]);
        elseif seq4(i) == 'T'
            x(i) = mean([x(i-1), T(1)]);
            y(i) = mean([y(i-1), T(2)]);
            z(i) = mean([z(i-1), T(3)]);
        elseif seq4(i) == 'C'
            x(i) = mean([x(i-1), C(1)]);
            y(i) = mean([y(i-1), C(2)]);
            z(i) = mean([z(i-1), C(3)]);
        elseif seq4(i) == 'G'
            x(i) = mean([x(i-1), G(1)]);
            y(i) = mean([y(i-1), G(2)]);
            z(i) = mean([z(i-1), G(3)]);
        end
    end
end