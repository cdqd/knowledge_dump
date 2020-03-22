#!/usr/bin/env bash

### Key notes
# shell does not care about types
# if script is not `source`d, a new shell is spawned to run the script
# ", $, `, and \ are still interpreted by the shell, even when they're in double quotes.
# >/dev/null 2>&1 directs any outputs or errors to the null device
# : always evaluates to True
# $@ is all parameters
# $# is the number of parameters the script was called with
# $? contains the exit value of the last run command
# backtick (``) = eval
# $(...) = eval
# := for default variable assignment
# | to pipe output into next command
# programs: grep for string match, cut for split & search, sed for replace
# be careful of scoping: assigning a variable inside a function assigns it globally (except for reserved variables).
# functions cannot change the variables passed to them

### Less important notes
# Note: use `shift` if we want to take more than 9 parameters
# $$ is the PID of the currently running shell
# $! is the PID of the last run background process
# $IFS = internal field separator.

### References
# https://www.shellscript.sh/quickref.html
# https://www.shellscript.sh/tips/

prompt_example() {
  echo "What is your name?"
  read MY_NAME  # read command automatically places quotes around its input
  echo "Hello $MY_NAME - hope you're well."
  touch "${MY_NAME}_generated_file" "${MY_NAME}_a"  # touch creates an empty file
  rm "${MY_NAME}"*
}

forloop_example() {
  for i in hello 1 * 2 goodbye
  do 
    echo "Looping ... i is set to $i."
  done
}

whileloop_example() {
  INPUT_STRING=hello
  while [ "$INPUT_STRING" != "bye" ]
  do
    echo "Please type something in (bye to quit)"
    read INPUT_STRING
    echo "You typed: $INPUT_STRING"
  done
}

test_example() {
  if [ 6 -eq 5 ]
  then echo "yes"
  elif [ 6 -ne 6 ]
  then echo "no"
  elif [ "ab" == "ab" ]
  then echo "ya"
  fi 
}

case_example() {
  echo "Please talk to me ..."
  while :
  do
    read INPUT_STRING
    case $INPUT_STRING in
    hello)
      echo "Hello yourself!"
      ;;
    bye)
      echo "See you again!"
      break
      ;;
    *)
      echo "Sorry, I don't understand"
      ;;
    esac
  done
  echo 
  echo "That's all folks!"
}

reserved_variables_example() {
  echo "n of function = $#"
  echo "base = `basename $0`"
  echo "first arg = $1"
  echo "exit status of last command: $?"
}

default_variables_example() {
  echo "What is your name? Hit Return to fetch default."
  read myname
  echo "Your name is: ${myname:=Cuong Duong}"
}

reserved_variables_example "ta" "tb" "tc"
