function bytes = get_file_size(fname)
% gets file size in bytes, ensuring that symlinks are dereferenced
% MP: not sure who wrote this code, but they were careful to do it right on Linux
    bytes = NaN;
    if isunix
        cmd = sprintf('stat -Lc %%s %s', fname);
        [status, r] = system(cmd);
        if status == 0
            bytes = str2double(r);
        end
    end

    if isnan(bytes)
        o = dir(fname);
        bytes = o.bytes;
    end
end
