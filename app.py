"""Measure how a neural morphology classifier changes across synthetic acquisition conditions."""
from kit import *
TITLE='Microscopy Morphology Under Domain Shift'
LIVE=False

def generate(seed,n,brightness):
    rng=np.random.default_rng(seed);y=rng.integers(0,2,n);images=[]
    for label in y:
        im=rng.normal(0,.08,(12,12));prob=.15 if label==0 else .55;im+=(rng.random((12,12))<prob)*.8;images.append(im*brightness+.3*(brightness-1))
    return np.array(images),y
def features(images,normalize):
    values=[]
    for im in images:
        if normalize:im=(im-im.mean())/max(im.std(),1e-8)
        values.append(np.r_[np.histogram(im,bins=8,range=(-2,3))[0]/im.size,im.mean(),im.std(),np.mean(abs(np.diff(im,axis=0))),np.mean(abs(np.diff(im,axis=1)))])
    return np.array(values)
def fit(x,y,steps=350):
    rng=np.random.default_rng(9);w1=rng.normal(0,.2,(x.shape[1],12));b1=np.zeros(12);w2=rng.normal(0,.2,(12,2));b2=np.zeros(2)
    for _ in range(steps):
        h=np.tanh(x@w1+b1);prob=softmax(h@w2+b2);delta=(prob-np.eye(2)[y])/len(y);dh=(delta@w2.T)*(1-h*h);g1=x.T@dh;gb1=dh.sum(axis=0);g2=h.T@delta;gb2=delta.sum(axis=0);w1-=.2*g1;b1-=.2*gb1;w2-=.2*g2;b2-=.2*gb2
    return w1,b1,w2,b2
def predict(m,x):return (np.tanh(x@m[0]+m[1])@m[2]+m[3]).argmax(axis=1)
def run(d,a):
    images,y=generate(a.seed,500,1.);test,truth=generate(a.seed+1,200,1.8);results={};art={}
    for normalize in (False,True):
        x=features(images,normalize);mean,scale=standardize_fit(x);m=fit((x-mean)/scale,y);pred=predict(m,(features(test,normalize)-mean)/scale);name='normalized' if normalize else 'raw';results[name]=classification(truth,pred)
        for i,v in enumerate(m):art[name+'_p'+str(i)]=v
    return dict(domain_results=results,majority_baseline_accuracy=float(np.mean(truth==np.bincount(y).argmax())),training_brightness=1.,test_brightness=1.8,labels='Fictional sparse/dense morphology; no diagnostic labels'),art

if __name__=="__main__":main(run)
