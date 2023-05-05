"""
/*********************************************************************
* Copyright (c) 2023 the Qrisp Authors
*
* This program and the accompanying materials are made
* available under the terms of the Eclipse Public License 2.0
* which is available at https://www.eclipse.org/legal/epl-2.0/
*
* SPDX-License-Identifier: EPL-2.0
**********************************************************************/
"""


import logging
import threading

from thrift.server import TServer


class StoppableThriftServer(TServer.TThreadedServer):
    def __init__(*args, **kwargs):
        args[0].is_running = False
        TServer.TThreadedServer.__init__(*args, **kwargs)

    def serve(self):
        self.serverTransport.listen()

        self.stop_flag = False
        thread_list = []
        while True:
            if self.stop_flag:
                for t in thread_list:
                    t.join()
                break
            try:
                self.is_running = True
                client = self.serverTransport.accept()
                t = threading.Thread(target=self.handle, args=(client,))
                t.setDaemon(self.daemon)
                t.start()
                thread_list.append(t)
            except OSError:
                pass
            except KeyboardInterrupt:
                raise
            except Exception as x:
                logging.exception(x)

    def stop(self):
        self.stop_flag = True
        try:
            self.serverTransport.handle.shutdown(2)
        except OSError:
            pass
        self.serverTransport.handle.close()
