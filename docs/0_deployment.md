# Deployment

## Docker

I decided to use docker for this submission, as it simplifies the setup required for the interviewer to run the submission.

In particular, I could not guarantee that the interviewer's system:

* was permissioned to install a new Python environment
* had a Python version compatible with the dependencies listed in this submission
* had access to some more obscure python dependencies used, such as `skl2onnx`
* contained the same version of some OS libraries used by Python dependencies, such as `libsqlite.so`

The following assumptions were made about the interviewer's system:

* runs an x86 CPU
* has permission to run docker containers
* has internet access to download my container from docker.io
