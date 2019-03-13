function bytes = get_file_size(fname)
% gets file size ensuring that symlinks are dereferenced
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
