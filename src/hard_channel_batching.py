"""Equal-exposure place packing; no validation labels, replacement or reweighting."""
import numpy as np

ARMS=('random_ms','random_protect','hard_ms','hard_protect')


def validate_partition(batches,count,width=16):
    if count<width or count%width or len(batches)!=count//width:
        raise ValueError('Invalid partition shape')
    if any(len(b)!=width or len(set(b))!=width or any(type(x) is not int for x in b) for b in batches):
        raise ValueError('Invalid/duplicate place within batch')
    flat=[p for b in batches for p in b]
    if sorted(flat)!=list(range(count)):raise ValueError('Every training place must appear exactly once')


def pack_places(neighbors,seed,fallback,width=16):
    neighbors=np.asarray(neighbors)
    if neighbors.ndim!=2 or not np.issubdtype(neighbors.dtype,np.integer):raise ValueError('Integer neighbor graph required')
    n=len(neighbors)
    if n<width or n%width or ((neighbors<0)|(neighbors>=n)).any():raise ValueError('Invalid graph')
    unused=set(range(n));result=[];fallback_batches=0
    for anchor in np.random.default_rng(seed).permutation(n).tolist():
        if anchor not in unused:continue
        group=[anchor];unused.remove(anchor)
        for candidate in neighbors[anchor].tolist():
            if candidate in unused:
                group.append(candidate);unused.remove(candidate)
                if len(group)==width:break
        if len(group)<width:
            fallback_batches+=1
            # Fallback is exact nearest among the remaining places, not random.
            for candidate in fallback(anchor,sorted(unused)):
                candidate=int(candidate)
                if candidate in unused:
                    group.append(candidate);unused.remove(candidate)
                    if len(group)==width:break
        if len(group)!=width:raise ValueError('Fallback did not complete the equal-exposure batch')
        result.append(group)
    validate_partition(result,n,width)
    return result,dict(fallback_batches=fallback_batches,batches=len(result),places=n)


def difficulty_gate(random_stats,hard_stats,count):
    minimum=int(np.ceil(count*.01))
    checks=dict(same_query_exposure=random_stats['queries']==hard_stats['queries']==count,
        all_queries_have_valid_negatives=random_stats['valid']==hard_stats['valid']==count,
        enough_low_margin_queries=hard_stats['low_margin']>=minimum,
        doubles_low_margin_queries=hard_stats['low_margin']>=2*max(1,random_stats['low_margin']),
        not_fewer_teacher_errors=hard_stats['errors']>=random_stats['errors'])
    return dict(checks=checks,pass_all=all(checks.values()),minimum_low_margin_queries=minimum)
