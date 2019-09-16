# -*- coding: utf-8 -*-
import numpy

"""Top-level package for Sequel."""

__author__ = """Johan Carlin"""
__email__ = 'johan.carlin@gmail.com'
__version__ = '0.1.0'

def getsequence(seqhand,nattempts=1000,*args,**kwargs):
    attempt = 0
    while attempt < nattempts:
        attempt += 1
        seq = seqhand(*args,**kwargs)
        if not numpy.any(numpy.isnan(seq)):
            return seq,attempt
    return numpy.nan,attempt

# insight 1: you can brute-force a first-order counter-balanced sequence
# quite easily even for fairly large condition numbers (<100 works fine)
def transferseq(tmat,startind=None):
    # important to avoid overlap across calls
    tmat = numpy.copy(tmat)
    ntransfers = numpy.sum(tmat.flat)
    dims = numpy.shape(tmat)
    assert len(dims)==2, 'input tmat must be 2D'
    assert dims[0] == dims[1], 'input tmat must be square matrix'
    ncon = dims[0]
    if startind is None:
        startind = numpy.random.randint(ncon)
    seq = [startind]
    niter = 0
    while numpy.sum(tmat.flat)>0.:
        niter += 1
        # the current trial is in rows
        rowind = seq[-1]
        validcol = numpy.where(tmat[rowind,:] > 0.)[0]
        if not len(validcol):
            return numpy.nan
        colind = validcol[numpy.random.randint(len(validcol))]
        seq.append(colind)
        tmat[rowind,colind] -= 1
    return seq


def transfermatrix(seq):
    ncon = numpy.max(seq).astype(int)+1
    out = numpy.zeros([ncon,ncon,len(seq)-1])
    out[seq[:-1],seq[1:],range(0,len(seq)-1)] = 1
    return numpy.sum(out,axis=2)

def permutationrep(x,n,allowrep=False,maxiter=1000):
    """generate repeating sequence of random indices in 0:x range."""
    seq = numpy.arange(x)
    numpy.random.shuffle(seq)
    niter = 0
    while len(seq) < (x*n):
        niter += 1
        newseq = numpy.arange(x)
        numpy.random.shuffle(newseq)
        if not seq[-1] == newseq[0] or allowrep:
            seq = numpy.concatenate([seq,newseq])
        assert niter <= maxiter, 'max iteration limit exceeded'
    return seq

def insertrep(seq,n,repcode=None,maxrep=numpy.inf):
    """insert repetitions in sequence seq."""
    u= numpy.unique(seq)
    if n==0:
        return seq
    lseq = [[x] for x in seq]
    nseq = numpy.zeros(len(lseq))
    assert not numpy.any(numpy.diff(seq)==0), \
            'need a non-repeating input sequence'
    for con in u:
        rep = 0
        while rep < n:
            # pick a random trial where we haven't already maxed out reps
            validind = numpy.where((seq==con) & (nseq<maxrep))[0]
            assert len(validind),\
                    'no remaining trials to repeat - too high n?'
            target = validind[numpy.random.randint(len(validind))]
            rep += 1
            repval = seq[target]
            if repcode:
                repval = repcode
            lseq[target].append(repval)
            nseq[target] += 1
    # unpack the list of lists and return
    return [l for subl in lseq for l in subl]
