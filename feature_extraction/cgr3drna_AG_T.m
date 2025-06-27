% DNA or RNA 3D CGR
function [x, y, z] = cgr3drna_AG_T(seq)
    A = ([0, 0, 0]-(1/2))*2*sqrt(1/3);
    T = ([1, 1, 0]-(1/2))*2*sqrt(1/3);
    U = ([1, 1, 0]-(1/2))*2*sqrt(1/3);
    G = ([1, 0, 1]-(1/2))*2*sqrt(1/3);
    C = ([0, 1, 0]-(1/2))*2*sqrt(1/3);
    if seq(1) == 'A'
        x(1) = mean([0, A(1)]);
        y(1) = mean([0, A(2)]);
        z(1) = mean([0, A(3)]);
    elseif seq(1) == 'T'
        x(1) = mean([0, T(1)]);
        y(1) = mean([0, T(2)]);
        z(1) = mean([0, T(3)]);
    elseif seq(1) == 'U'
        x(1) = mean([0, U(1)]);
        y(1) = mean([0, U(2)]);
        z(1) = mean([0, U(3)]);
    elseif seq(1) == 'C'
        x(1) = mean([0, C(1)]);
        y(1) = mean([0, C(2)]);
        z(1) = mean([0, C(3)]);
    elseif seq(1) == 'G'
        x(1) = mean([0, G(1)]);
        y(1) = mean([0, G(2)]);
        z(1) = mean([0, G(3)]);
    end
    for i = 2:length(seq)
        if seq(i) == 'A'
            x(i) = mean([x(i-1), A(1)]);
            y(i) = mean([y(i-1), A(2)]);
            z(i) = mean([z(i-1), A(3)]);
        elseif seq(i) == 'T'
            x(i) = mean([x(i-1), T(1)]);
            y(i) = mean([y(i-1), T(2)]);
            z(i) = mean([z(i-1), T(3)]);
        elseif seq(i) == 'U'
            x(i) = mean([x(i-1), U(1)]);
            y(i) = mean([y(i-1), U(2)]);
            z(i) = mean([z(i-1), U(3)]);
        elseif seq(i) == 'C'
            x(i) = mean([x(i-1), C(1)]);
            y(i) = mean([y(i-1), C(2)]);
            z(i) = mean([z(i-1), C(3)]);
        elseif seq(i) == 'G'
            x(i) = mean([x(i-1), G(1)]);
            y(i) = mean([y(i-1), G(2)]);
            z(i) = mean([z(i-1), G(3)]);
        end
    end
end