include subtrees/z_quantum_actions/Makefile

github_actions:
	python3 -m venv my_little_venv && \
		my_little_venv/bin/python3 -m pip install --upgrade pip && \
		my_little_venv/bin/python3 -m pip install -e orquestra-quantum && \
		my_little_venv/bin/python3 -m pip install -e '.[dev]'