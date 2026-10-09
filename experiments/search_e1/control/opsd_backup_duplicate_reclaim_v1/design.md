# Fixed old-backup duplicate reclamation design

The authorized choice retains the existing accepted archives byte for byte and
reclaims only their76 verified loose duplicate files. Recompressing or deleting
archives, changing checkpoint serialization, removing directories and touching
Node originals are outside this helper. The existing GNU protected archive is
handled independently from the CP89 USTAR archive and the unrelated strict
USTAR compact checkpoint queue.

Prepare binds known historical receipt/inventory SHA, exact fixed roots/run and
archive identity. One bounded archive pass fresh verifies whole and member SHA;
one loose-file pass fresh verifies every corresponding member SHA. An immutable
plan records every current regular-file identity and all preserved identities.
Execute requires that exact plan and source publication, checks current small
metadata, rejects open or foreign files, and writes durable operation intent.
Each file has its own fsynced pending and completion journal; explicit pending
allows honest recovery from a crash between unlink and completion write.

All archive/receipt paths remain intact. Current inode/time identity and lsof
observations reduce race exposure; they cannot establish a global no-writer
condition against unrelated noncooperating processes. Root schedules this exact
scope when quiescent. The two local leases coordinate this helper's callers.
The actual GPU scientific queue never waits on prepare, execute or backup space.
