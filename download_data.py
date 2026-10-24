"""Authorized explicit dataset downloads only, with published SHA256 verification."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
SOURCES={
 'mnist':('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz','train-images-idx3-ubyte.gz','440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609'),
 'official_test':('https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy','mnist_test_seq.npy','c2a3e8d3939c001ab77bf543f130157424efa486bb4d0974d2cebb773b13f8b3')}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--mnist',action='store_true');p.add_argument('--official-test',action='store_true');a=p.parse_args();Path('data').mkdir(exist_ok=True)
 for key in ['mnist','official_test']:
  if not getattr(a,key):continue
  url,name,expected=SOURCES[key];target=Path('data')/name
  if not target.exists():
   with urllib.request.urlopen(url,timeout=60) as response,target.open('wb') as output:
    while True:
     block=response.read(1048576)
     if not block:break
     output.write(block)
  h=hashlib.sha256()
  with target.open('rb') as stream:
   for block in iter(lambda:stream.read(1048576),b''):h.update(block)
  if h.hexdigest()!=expected:raise RuntimeError('Incomplete/corrupt dataset; checksum mismatch, do not use')
  metadata={'url':url,'sha256':expected,'bytes':target.stat().st_size,'verified':True}
  Path('data', 'test_source.json' if key=='official_test' else 'mnist_source.json').write_text(json.dumps(metadata,indent=2)+'\n')
