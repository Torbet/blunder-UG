#!/bin/bash

gunicorn -w 4 -b 0.0.0.0:443 --certfile=/etc/letsencrypt/live/chess.torbet.co/fullchain.pem --keyfile=/etc/letsencrypt/live/chess.torbet.co/privkey.pem 'serve:app'