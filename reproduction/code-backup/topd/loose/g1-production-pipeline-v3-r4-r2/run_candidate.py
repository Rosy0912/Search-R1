"""exec the actual G1 source from its own directory, never parent TOPD repo cwd."""
import argparse,os,pathlib,sys
ROOT=pathlib.Path('/dataset/shaoshitong/search-r1-rl/topd')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--directory',required=True);ap.add_argument('command',nargs=argparse.REMAINDER);a=ap.parse_args()
 directory=pathlib.Path(a.directory).resolve()
 if not directory.is_relative_to(ROOT/'tools') or not (directory/'verl/trainer/main_ppo_eco.py').is_file():raise ValueError('actual isolated G1 source required')
 command=a.command[1:] if a.command[:1]==['--'] else a.command
 if command[:3]!=[str(ROOT/'env/bin/python'),'-m','verl.trainer.main_ppo_eco']:raise ValueError('unexpected actor entry')
 os.chdir(directory);os.execv(command[0],command)
if __name__=='__main__':main()
