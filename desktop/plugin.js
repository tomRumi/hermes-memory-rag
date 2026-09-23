/**
 * Hermes Memory — what the memory store holds, with buttons that do something.
 *
 *   <hermes home>/plugins/hermes-memory-rag/desktop/plugin.js
 *
 * The DESKTOP HALF of the plugin package. It draws a chip in the status bar and
 * a panel behind it:
 *
 *   - what each project has stored, how many notes are waiting to be merged into
 *     the written pages, how many pages and facts there are
 *   - when the last backup ran, and whether pushing it to a remote is on
 *   - Backup now / Export / Import / Claude Code import / Review
 *   - the push setting, which is off until it is turned on here
 *
 * It does not touch the memory store itself. The window cannot run commands, and
 * it must not: the store's libraries are deliberately kept out of Hermes's own
 * environment. Every button asks the plugin's backend routes
 * (/api/plugins/hermes-memory-rag/...) which run the project's scripts in the
 * environment they were installed into.
 *
 * Plain ESM, loaded uncompiled — jsx() calls only, no JSX syntax. Only
 * @hermes/plugin-sdk, react and react/jsx-runtime resolve. Class names below are
 * ones the app's own compiled stylesheet already emits; a class it does not emit
 * would silently do nothing.
 */
import { Button, cn, Codicon, Dialog, DialogContent, haptic, host, Tip, useValue } from '@hermes/plugin-sdk'
import { jsx, jsxs, Fragment } from 'react/jsx-runtime'
import { useCallback, useEffect, useState } from 'react'

const ID = 'hermes-memory-rag'

// Captured in register(): component code cannot see ctx.
let rest = null
let osApi = null

async function ask(path, body) {
  if (!rest) throw new Error('plugin not registered')
  return rest(path, body === undefined
    ? undefined
    : { method: 'POST', body: JSON.stringify(body), headers: { 'content-type': 'application/json' } })
}

function fmtWhen(iso) {
  if (!iso) return 'never'
  const then = new Date(iso.endsWith('Z') ? iso : `${iso}Z`)
  if (Number.isNaN(then.getTime())) return iso
  const minutes = Math.round((Date.now() - then.getTime()) / 60000)
  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${minutes} min ago`
  if (minutes < 60 * 24) return `${Math.round(minutes / 60)} h ago`
  return `${Math.round(minutes / (60 * 24))} d ago`
}

function MemoryChip() {
  const [open, setOpen] = useState(false)
  const [status, setStatus] = useState(null)
  const [failed, setFailed] = useState(false)

  const refresh = useCallback(async () => {
    try {
      setStatus(await ask('/status'))
      setFailed(false)
    } catch {
      setFailed(true)
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const nodes = status
    ? (status.projects || []).reduce((sum, p) => sum + (p.code || 0) + (p.wiki || 0) + (p.memory || 0), 0)
    : 0
  const label = failed ? 'memory —' : `memory ${nodes}`
  const tooltip = failed
    ? 'Memory: the plugin backend did not answer'
    : `Memory: ${nodes} stored items, ${status?.wiki_pages ?? 0} wiki pages, ${status?.facts ?? 0} facts · last backup ${fmtWhen(status?.last_backup?.started)}`

  return jsxs(Fragment, {
    children: [
      jsx(Tip, {
        label: tooltip,
        children: jsx('button', {
          type: 'button',
          'aria-label': tooltip,
          className: cn(
            'inline-flex h-full items-center gap-1 whitespace-nowrap px-1.5 text-[0.6875rem] tabular-nums transition-colors',
            failed
              ? 'text-destructive'
              : 'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground'
          ),
          onClick: () => { haptic('tap'); setOpen(true) },
          children: label
        })
      }),
      jsx(MemoryDialog, { open, onOpenChange: setOpen, status, refresh, failed })
    ]
  })
}

function Row({ label, value, tone }) {
  return jsxs('div', {
    className: 'flex items-baseline justify-between gap-3',
    children: [
      jsx('span', { className: 'text-[0.6875rem] text-muted-foreground', children: label }),
      jsx('span', {
        className: cn('text-[0.6875rem] tabular-nums',
          tone === 'bad' ? 'text-destructive' : 'text-foreground'),
        children: value
      })
    ]
  })
}

function MemoryDialog({ open, onOpenChange, status, refresh, failed }) {
  const [busy, setBusy] = useState('')
  const [note, setNote] = useState(null)
  const [confirmImport, setConfirmImport] = useState(false)

  const run = useCallback(async (what, path, body) => {
    setBusy(what)
    setNote(null)
    haptic('tap')
    try {
      const result = await ask(path, body)
      const text = result?.ok
        ? 'done'
        : (result?.error || result?.output || 'did not finish')
      setNote({ kind: result?.ok ? 'ok' : 'error', text: String(text).split('\n').slice(-1)[0].slice(0, 300) })
      await refresh()
    } catch (error) {
      setNote({ kind: 'error', text: String(error?.message || error).slice(0, 300) })
    } finally {
      setBusy('')
    }
  }, [refresh])

  const exportNow = useCallback(async () => {
    if (!osApi?.pickSavePath) {
      setNote({ kind: 'error', text: 'this build has no file picker; export from the command line' })
      return
    }
    const path = await osApi.pickSavePath({ title: 'Save memory export', defaultPath: 'memory-export.jsonl' })
    if (!path) return
    await run('export', '/export', { path })
  }, [run])

  const importFrom = useCallback(async () => {
    if (!confirmImport) {
      setConfirmImport(true)
      setNote({ kind: 'info', text: 'Importing replaces what is stored now. Press again to choose a file.' })
      return
    }
    setConfirmImport(false)
    if (!osApi?.pickOpenPath) {
      setNote({ kind: 'error', text: 'this build has no file picker; import from the command line' })
      return
    }
    const path = await osApi.pickOpenPath({ title: 'Choose a memory export (.jsonl)' })
    if (!path) return
    await run('import', '/import', { path })
  }, [run, confirmImport])

  const togglePush = useCallback(async () => {
    const next = !(status?.push_enabled)
    setBusy('push')
    try {
      await ask('/push', { enabled: next })
      setNote({ kind: 'ok', text: next
        ? 'Pushing to the remote is on. The backup will push from now on.'
        : 'Pushing is off. Backups stay on this machine.' })
      await refresh()
    } catch (error) {
      setNote({ kind: 'error', text: String(error?.message || error).slice(0, 200) })
    } finally {
      setBusy('')
    }
  }, [status, refresh])

  const projects = status?.projects || []
  const last = status?.last_backup

  return jsx(Dialog, {
    open,
    onOpenChange,
    children: jsx(DialogContent, {
      className: 'w-[23rem]',
      children: jsxs('div', {
        className: 'flex flex-col gap-3.5 p-1 text-[0.75rem]',
        children: [
          jsxs('div', {
            className: 'flex items-baseline justify-between gap-2',
            children: [
              jsx('p', { className: 'font-medium text-foreground', children: 'Memory' }),
              jsx('span', { className: 'text-[0.6875rem] text-muted-foreground', children: 'stored locally' })
            ]
          }),

          failed
            ? jsx('p', { className: 'text-[0.6875rem] text-destructive',
                         children: 'The plugin backend did not answer. Is it enabled in Settings → Plugins?' })
            : jsxs('div', {
                className: 'flex flex-col gap-1',
                children: [
                  jsx('p', { className: 'text-[0.6875rem] font-medium uppercase tracking-wide text-(--ui-text-quaternary)', children: 'Stored' }),
                  projects.length === 0
                    ? jsx('p', { className: 'text-[0.6875rem] text-muted-foreground', children: 'Nothing indexed yet.' })
                    : projects.map(p => jsx(Row, {
                        label: p.project,
                        value: `${p.code || 0} code · ${p.wiki || 0} wiki · ${p.memory || 0} notes`,
                        tone: null
                      }, p.project)),
                  jsx(Row, { label: 'wiki pages on disk', value: `${status?.wiki_pages ?? 0}` }),
                  jsx(Row, { label: 'facts (recalled each turn)', value: `${status?.facts ?? 0}` })
                ]
              }),

          jsxs('div', {
            className: 'flex flex-col gap-1',
            children: [
              jsx('p', { className: 'text-[0.6875rem] font-medium uppercase tracking-wide text-(--ui-text-quaternary)', children: 'Backup' }),
              jsx(Row, { label: 'last run', value: fmtWhen(last?.started), tone: last?.started ? null : 'bad' }),
              last?.committed !== undefined && jsx(Row, { label: 'paths committed', value: `${last.committed}` }),
              jsx(Row, { label: 'push to remote', value: status?.push_enabled ? 'on' : 'off' })
            ]
          }),

          jsxs('div', {
            className: 'flex flex-col gap-1.5',
            children: [
              jsx(Button, {
                size: 'sm',
                variant: 'default',
                disabled: !!busy,
                onClick: () => run('backup', '/backup'),
                children: busy === 'backup' ? 'Backing up…' : 'Backup now'
              }),
              jsxs('div', {
                className: 'flex items-center justify-between gap-3',
                children: [
                  jsx(Button, { size: 'sm', variant: 'ghost', disabled: !!busy,
                                onClick: exportNow, children: 'Export…' }),
                  jsx(Button, { size: 'sm', variant: 'ghost', disabled: !!busy,
                                onClick: importFrom,
                                children: confirmImport ? 'Confirm import…' : 'Import…' })
                ]
              }),
              jsxs('div', {
                className: 'flex items-center justify-between gap-3',
                children: [
                  jsx(Button, { size: 'sm', variant: 'ghost', disabled: !!busy,
                                onClick: () => run('claude', '/claude-import'),
                                children: busy === 'claude' ? 'Reading…' : 'Claude Code memory' }),
                  jsx(Button, { size: 'sm', variant: 'ghost', disabled: !!busy,
                                onClick: () => run('review', '/review'),
                                children: busy === 'review' ? 'Checking…' : 'Review' })
                ]
              }),
              jsx(Button, {
                size: 'sm',
                variant: 'ghost',
                disabled: !!busy,
                onClick: togglePush,
                children: busy === 'push' ? 'Saving…' : (status?.push_enabled ? 'Turn push off' : 'Turn push on')
              }),
              jsx('p', {
                className: 'text-[0.6875rem] text-muted-foreground',
                children: 'Import replaces what is stored now. Everything stays on this machine unless pushing is on.'
              })
            ]
          }),

          note && jsx('p', {
            className: cn('text-[0.6875rem]',
              note.kind === 'error' ? 'text-destructive'
                : note.kind === 'ok' ? 'text-foreground' : 'text-muted-foreground'),
            children: note.text
          }),

          jsx('div', {
            className: 'flex justify-end',
            children: jsx(Button, { size: 'sm', variant: 'ghost', onClick: () => onOpenChange(false), children: 'Close' })
          })
        ]
      })
    })
  })
}

export default {
  id: ID,
  name: 'Memory',
  register(ctx) {
    rest = typeof ctx?.rest === 'function' ? ctx.rest : null
    osApi = ctx?.os || null
    ctx.register({
      id: 'chip',
      area: 'statusBar.right',
      order: 340,
      render: () => jsx(MemoryChip, {})
    })
  }
}