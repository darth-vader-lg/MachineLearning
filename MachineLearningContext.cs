﻿using MachineLearning.TensorFlow;
using Microsoft.ML;
using Microsoft.ML.Runtime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace MachineLearning
{
   /// <summary>
   /// Contesto di machine learning
   /// </summary>
   [Serializable]
   public class MachineLearningContext : IContextProvider<MLContext>, IContextProvider<TFContext>, IDeserializationCallback, IDisposable
   {
      #region Fields
      /// <summary>
      /// Tutti gli elementi IDisposable dei contesti
      /// </summary>
      private static readonly HashSet<MachineLearningContext> _allContexts = new();
      /// <summary>
      /// Scheduler di creazione dell'oggetto
      /// </summary>
      [NonSerialized]
      private TaskScheduler _creationTaskScheduler;
      /// <summary>
      /// Thread di creazione dell'oggetto
      /// </summary>
      [NonSerialized]
      private Thread _creationThread;
      /// <summary>
      /// Elementi IDisposable del contesto
      /// </summary>
      [NonSerialized]
      private readonly HashSet<IDisposable> _disposables = new();
      /// <summary>
      /// Oggetto disposto
      /// </summary>
      private bool _disposedValue;
      /// <summary>
      /// Coda dei messaggi di log
      /// </summary>
      [NonSerialized]
      private Queue<MachineLearningLogEventArgs> _logMessages;
      /// <summary>
      /// Seme per le operazioni random
      /// </summary>
      private readonly int? _seed;
      /// <summary>
      /// Task di lavoro del contesto
      /// </summary>
      [NonSerialized]
      private readonly HashSet<(Task task, CancellationTokenSource cts)> _workingTasks = new();
      #endregion
      #region Properties
      /// <summary>
      /// Determina se generare i messaggi di log in maniera asincrona o sincrona con il thread di creazione del contesto
      /// </summary>
      public bool SyncLogs { get; set; }
      /// <summary>
      /// Contesto di default
      /// </summary>
      public static MachineLearningContext Default { get; } = new MachineLearningContext();
      /// <summary>
      /// Descrizione contesto ML.NET
      /// </summary>
      string IExceptionContext.ContextDescription => ((IExceptionContext)MLNET).ContextDescription;
      /// <summary>
      /// Contesto ML.NET
      /// </summary>
      MLContext IContextProvider<MLContext>.Context => MLNET;
      /// <summary>
      /// Contesto ML.NET
      /// </summary>
      TFContext IContextProvider<TFContext>.Context => TensorFlow;
      /// <summary>
      /// Contesto ML.NET
      /// </summary>
      [field: NonSerialized]
      public MLContext MLNET { get; private set; }
      /// <summary>
      /// Indica necessita' di postare un azione nel thread di creazione dal momento che ci si trova in un altro
      /// </summary>
      public bool PostRequired => Thread.CurrentThread != _creationThread && _creationTaskScheduler != null;
      /// <summary>
      /// Contesto TensorFlow
      /// </summary>
      [field: NonSerialized]
      public TFContext TensorFlow { get; private set; }
      #endregion
      #region Events
      /// <summary>
      /// Evento di log
      /// </summary>
      public event MachineLearningLogEventHandler Log;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="seed">Seme per le operazioni random</param>
      public MachineLearningContext(int? seed = null)
      {
         lock (_allContexts)
            _allContexts.Add(this);
         _seed = seed;
         OnInit();
      }
      /// <summary>
      /// Aggiunge all'elenco dei disposables
      /// </summary>
      /// <param name="disposable">Oggetto disposable</param>
      /// <returns>Il task</returns>
      internal IDisposable AddDisposable(IDisposable disposable)
      {
         lock (_disposables) {
            _disposables.Add(disposable);
            return disposable;
         }
      }
      /// <summary>
      /// Aggiunge all'elenco un task di lavoro
      /// </summary>
      /// <param name="task"></param>
      /// <param name="cancellation">Token di cancellazione</param>
      /// <returns>Il task</returns>
      internal Task AddWorkingTask(Task task, CancellationTokenSource cancellation)
      {
         lock (_workingTasks) {
            _workingTasks.Add((task, cancellation));
            return task;
         }
      }
      /// <summary>
      /// Asserisce se un contesto e' valido
      /// </summary>
      /// <param name="provider">Il provider</param>
      /// <param name="name">Nome del parametro</param>
      public static void AssertContext<T>(IContextProvider<T> provider, string name) where T : class
      {
         Contracts.AssertValue(provider, name);
         Contracts.AssertValue(provider.Context, $"{name}.{nameof(IContextProvider<T>.Context)}");
      }
      /// <summary>
      /// Verifica che un provider di contesto sia valido per l'ML.NET
      /// </summary>
      /// <param name="provider">Il provider</param>
      /// <param name="name">Nome del parametro</param>
      public static void CheckContext<T>(IContextProvider<T> provider, string name) where T : class
      {
         Contracts.CheckValue(provider, name);
         Contracts.CheckValue(provider.Context, $"{name}.{nameof(IContextProvider<T>.Context)}");
      }
      /// <summary>
      /// Implementazione della IDisposable
      /// </summary>
      public void Dispose()
      {
         Dispose(disposing: true);
         GC.SuppressFinalize(this);
      }
      /// <summary>
      /// Funzione di dispose
      /// </summary>
      /// <param name="disposing">Indicatore di dispose da programma</param>
      protected virtual void Dispose(bool disposing)
      {
         if (!_disposedValue) {
            Stop(disposing ? -1 : 10000);
            _disposables.All(d =>
            {
               try {
                  d.Dispose();
               }
               catch (Exception exc) {
                  Trace.WriteLine(exc);
               }
               return true;
            });
            _disposedValue = true;
         }
      }
      /// <summary>
      /// Dispose di tutti i contesti
      /// </summary>
      public static void DisposeAll()
      {
         lock (_allContexts) {
            var tasks = (from c in _allContexts select Task.Run(() => c.Dispose())).ToArray();
            Task.WaitAll(tasks);
         }
      }
      /// <summary>
      /// Avvia un canale di messaggistica
      /// </summary>
      /// <param name="name">Nome del canale</param>
      /// <returns>Il canale</returns>
      IChannel IChannelProvider.Start(string name) => ((IChannelProvider)MLNET).Start(name);
      /// <summary>
      /// Avvia una pipe di messaggistica
      /// </summary>
      /// <typeparam name="TMessage">Tipo di messaggio</typeparam>
      /// <param name="name">Nome della pipe</param>
      /// <returns>La pipe</returns>
      IPipe<TMessage> IChannelProvider.StartPipe<TMessage>(string name) => ((IChannelProvider)MLNET).StartPipe<TMessage>(name);
      /// <summary>
      /// Processa un'eccezione
      /// </summary>
      /// <typeparam name="TException">Tipo di eccezione</typeparam>
      /// <param name="ex">Eccezione</param>
      /// <returns>L'eccezione processata</returns>
      TException IExceptionContext.Process<TException>(TException ex) => ((IExceptionContext)MLNET).Process(ex);
      /// <summary>
      /// Log della ML.NET
      /// </summary>
      /// <param name="sender">Contesto ML.NET</param>
      /// <param name="e">Argomenti del log</param>
      private void NET_Log(object sender, Microsoft.ML.LoggingEventArgs e)
      {
         var kind = e.Kind switch
         {
            ChannelMessageKind.Info => MachineLearningLogKind.Info,
            ChannelMessageKind.Warning => MachineLearningLogKind.Warning,
            ChannelMessageKind.Error => MachineLearningLogKind.Error,
            _ => MachineLearningLogKind.Trace,
         };
         if (!SyncLogs) {
            OnLog(new MachineLearningLogEventArgs(e.Message, kind, e.Source));
            return;
         }
         lock (_logMessages ??= new Queue<MachineLearningLogEventArgs>()) {
            _logMessages.Enqueue(new MachineLearningLogEventArgs(e.Message, kind, e.Source));
            if (_logMessages.Count == 1) {
               Post(() =>
               {
                  for (; ; ) {
                     lock (_logMessages) {
                        if (_logMessages.Count == 0)
                           break;
                        OnLog(_logMessages.Dequeue());
                     }
                  }
               });
            }
         }
      }
      /// <summary>
      /// Funzione post deserializzazione
      /// </summary>
      /// <param name="sender"></param>
      void IDeserializationCallback.OnDeserialization(object sender) => OnInit();
      /// <summary>
      /// Funzione di inizializzazione
      /// </summary>
      protected virtual void OnInit()
      {
         // Memorizza lo scheduler e il thread di creazione
         _creationThread = Thread.CurrentThread;
         if (SynchronizationContext.Current != null)
            _creationTaskScheduler = TaskScheduler.FromCurrentSynchronizationContext();
         // Inizializza il contesto ML.NET
         MLNET = new MLContext(_seed);
         MLNET.Log += NET_Log;
         // Inizializza il contesto TensorFlow
         TensorFlow = new TFContext();
         TensorFlow.Log += TensorFlow_Log;
      }
      /// <summary>
      /// Funzione di log di un messaggio
      /// </summary>
      /// <param name="e">Argomenti del messaggio</param>
      protected virtual void OnLog(MachineLearningLogEventArgs e)
      {
         try {
            Log?.Invoke(this, e);
         }
         catch (Exception exc) {
            Trace.WriteLine(exc);
         }
      }
      /// <summary>
      /// Posta un azione nel thread di creazione del contesto
      /// </summary>
      /// <param name="Action">Azione</param>
      public void Post(Action Action)
      {
         if (PostRequired)
            new Task(Action).Start(_creationTaskScheduler);
         else
            Action();
      }
      /// <summary>
      /// Rimuove dall'elenco dei disposables
      /// </summary>
      /// <param name="disposable">L'oggetto disposable</param>
      internal void RemoveDisposable(IDisposable disposable)
      {
         lock (_disposables)
            _disposables.Remove(disposable);
      }
      /// <summary>
      /// Rimuove dall'elenco un task di lavoro
      /// </summary>
      /// <param name="task">Il task</param>
      internal void RemoveWorkingTask(Task task)
      {
         lock (_workingTasks)
            _workingTasks.RemoveWhere(t => t.task == task);
      }
      /// <summary>
      /// Effettua lo stop di tutti i task del contesto
      /// </summary>
      /// <param name="timeoutMs">timeout</param>
      public void Stop(int timeoutMs = -1)
      {
         lock (_workingTasks) {
            Task.Run(() =>
            {
               _workingTasks.All(t => { t.cts.Cancel(); return true; });
               _workingTasks.All(t =>
               {
                  try {
                     t.task.Wait();
                  }
                  catch (Exception exc) {
                     Trace.WriteLine(exc);
                  }
                  return true;
               });
            }).Wait(timeoutMs);
            _workingTasks.Clear();
         }
      }
      /// <summary>
      /// Effettua lo stop di tutti i task dei contesti
      /// </summary>
      /// <param name="timeoutMs">timeout</param>
      public static void StopAll(int timeoutMs = -1)
      {
         lock (_allContexts) {
            var tasks = (from c in _allContexts select Task.Run(() => c.Stop(timeoutMs))).ToArray();
            Task.WaitAll(tasks, timeoutMs);
         }
      }
      /// <summary>
      /// Log della TensorFlow
      /// </summary>
      /// <param name="sender">Contesto ML.NET</param>
      /// <param name="e">Argomenti del log</param>
      private void TensorFlow_Log(object sender, TensorFlow.LoggingEventArgs e)
      {
         var kind = e.Kind switch
         {
            ChannelMessageKind.Info => MachineLearningLogKind.Info,
            ChannelMessageKind.Warning => MachineLearningLogKind.Warning,
            ChannelMessageKind.Error => MachineLearningLogKind.Error,
            _ => MachineLearningLogKind.Trace,
         };
         if (!SyncLogs) {
            OnLog(new MachineLearningLogEventArgs(e.Message, kind, e.Source));
            return;
         }
         lock (_logMessages ??= new Queue<MachineLearningLogEventArgs>()) {
            _logMessages.Enqueue(new MachineLearningLogEventArgs(e.Message, kind, e.Source));
            if (_logMessages.Count == 1) {
               Post(() =>
               {
                  for (; ; ) {
                     lock (_logMessages) {
                        if (_logMessages.Count == 0)
                           break;
                        OnLog(_logMessages.Dequeue());
                     }
                  }
               });
            }
         }
      }
      #endregion
   }
}
