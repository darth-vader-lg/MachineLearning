using Microsoft.ML;
using Microsoft.ML.Runtime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Tensorflow;
using MachineLearning.TensorFlow;

namespace MachineLearning
{
   /// <summary>
   /// Contesto di machine learning
   /// </summary>
   [Serializable]
   public class MachineLearningContext : IContextProvider<MLContext>, IContextProvider<TFContext>, IDeserializationCallback
   {
      #region Fields
      /// <summary>
      /// Tutti i task di lavoro
      /// </summary>
      private static readonly HashSet<List<(Task task, CancellationTokenSource cts)>> _allWorkingTasks = new();
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
      private readonly List<(Task task, CancellationTokenSource cts)> _workingTasks = new();
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
         _seed = seed;
         OnInit();
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
            lock (_allWorkingTasks)
               _allWorkingTasks.Add(_workingTasks);
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
      /// Rimuove dall'elenco un task di lavoro
      /// </summary>
      /// <param name="task">Il task</param>
      internal void RemoveWorkingTask(Task task)
      {
         lock (_workingTasks) {
            var ix = _workingTasks.FindIndex(t => t.task == task);
            _workingTasks.RemoveAt(ix);
            if (_workingTasks.Count == 0) {
               lock (_allWorkingTasks)
                  _allWorkingTasks.Remove(_workingTasks);
            }
         }
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
               _workingTasks.ForEach(t => t.cts.Cancel());
               _workingTasks.ForEach(t => t.task.Wait());
            }).Wait(timeoutMs);
            _workingTasks.Clear();
            lock (_allWorkingTasks)
               _allWorkingTasks.Remove(_workingTasks);
         }
      }
      /// <summary>
      /// Effettua lo stop di tutti i task dei contesti
      /// </summary>
      /// <param name="timeoutMs">timeout</param>
      public static void StopAll(int timeoutMs = -1)
      {
         lock (_allWorkingTasks) {
            Task.Run(() =>
            {
               foreach (var wt in _allWorkingTasks) {
                  lock (wt) {
                     foreach (var (task, cts) in wt)
                        cts.Cancel();
                  }
               }
               foreach (var wt in _allWorkingTasks) {
                  lock (wt) {
                     foreach (var (task, cts) in wt) {
                        if (cts.IsCancellationRequested)
                           task.Wait();
                     }
                  }
               }
            }).Wait(timeoutMs);
         }
      }

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
