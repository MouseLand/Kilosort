function rez = correct_time(rez)

ops = rez.ops;
NT = ops.NT;
deleted_batches = ops.deleted_batches;
st3 = rez.st3;

for q=1:numel(deleted_batches)
    entry_point=deleted_batches(q)*NT-NT;
    st3(st3(:,1)>entry_point,1)=st3(st3(:,1)>entry_point,1)+NT;
end

rez.st3=st3; 