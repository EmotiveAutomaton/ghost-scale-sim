"""Strong direct public-law compilers for the explicit G5 construction tasks."""


def board(public_world,target):
    if public_world['family']!='board' or not 0<=target<16 or target.bit_count()>3:raise ValueError('unadmitted direct board task')
    return [cell for cell in range(4) if target&(1<<cell)]


def assembly(public_world,target):
    if len(target)!=3 or any(x not in (0,1) for x in target):raise ValueError('complete binary assembly target required')
    pending=set(range(3));attached=set();program=[]
    while pending:
        ready=sorted(i for i in pending if public_world['parents'][i]<0 or public_world['parents'][i] in attached)
        if not ready:raise ValueError('cyclic or unavailable public support')
        part=ready[0];program.append(part)
        if public_world['defaults'][part]!=target[part]:program.append(6+part)
        attached.add(part);pending.remove(part)
    return program+[9]
