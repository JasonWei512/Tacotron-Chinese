import argparse
import os
import re
import tensorflow as tf
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer


# sentences = [
#   # From July 8, 2017 New York Times:
#   'Scientists at the CERN laboratory say they have discovered a new particle.',
#   'There’s a way to measure the acute emotional intelligence that has never gone out of style.',
#   'President Trump met with other leaders at the Group of 20 conference.',
#   'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
#   # From Google's Tacotron example page:
#   'Generative adversarial network or variational auto-encoder.',
#   'The buses aren\'t the problem, they actually provide a solution.',
#   'Does the quick brown fox jump over the lazy dog?',
#   'Talib Kweli confirmed to AllHipHop that he will be releasing an album in the next year.',
# ]

sentences = [
  'ni3 men5 you3 yi1 ge4 hao3',
  'quan2 shi4 jie4 pao3 dao4 shen2 me5 di4 fang1',
  'ni3 men5 bi3 qi2 ta1 de5 xi1 fang1 ji4 zhe3 pao3 de5 hai2 kuai4',
  'dan4 shi4 wen4 lai2 wen4 qu4 de5 wen4 ti2 dou1 tu1 sen1 po4',
  'sang1 tai4 na2 yi4 fu5',
  'gou3 li4 guo2 jia1 sheng1 si3 yi3',
  'qi3 yin1 huo4 fu2 bi4 qu1 zhi1'
]

def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(ckpt_dir):
  checkpoint = tf.train.get_checkpoint_state(ckpt_dir).model_checkpoint_path
  print(hparams_debug_string())
  synth = Synthesizer()
  synth.load(checkpoint)
  base_path = get_output_base_path(checkpoint)
  for i, text in enumerate(sentences):
    path = '%s-%03d.wav' % (base_path, i)
    print('Synthesizing: %s' % path)
    with open(path, 'wb') as f:
      f.write(synth.synthesize(text))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', default='logs-tacotron', help='Path to model checkpoint')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  hparams.parse(args.hparams)
  run_eval(args.checkpoint)


if __name__ == '__main__':
  main()
