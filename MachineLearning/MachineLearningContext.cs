using Microsoft.ML;
using Microsoft.ML.Runtime;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace MachineLearning
{
   /// <summary>
   /// Contesto di machine learning
   /// </summary>
   [Serializable]
   public class MachineLearningContext : IMachineLearningContextProvider
   {
      #region Fields
      /// <summary>
      /// Tutti i task di lavoro
      /// </summary>
      private static readonly HashSet<List<(Task task, CancellationTokenSource cts)>> _allWorkingTasks = new HashSet<List<(Task task, CancellationTokenSource cts)>>();
      /// <summary>
      /// Contesto ML.NET
      /// </summary>
      [NonSerialized]
      private MLContext _net;
      /// <summary>
      /// Seme per le operazioni random
      /// </summary>
      private readonly int? _seed;
      /// <summary>
      /// Task di lavoro del contesto
      /// </summary>
      [NonSerialized]
      private readonly List<(Task task, CancellationTokenSource cts)> _workingTasks = new List<(Task task, CancellationTokenSource cts)>();
      #endregion
      #region Properties
      /// <summary>
      /// Contesto di default
      /// </summary>
      public static MachineLearningContext Default { get; } = new MachineLearningContext();
      /// <summary>
      /// Contesto di machine learning
      /// </summary>
      MachineLearningContext IMachineLearningContextProvider.ML => this;
      /// <summary>
      /// Contesto ML.NET
      /// </summary>
      public MLContext NET { get => _net ??= new MLContext(_seed); private set { _net = value; } }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="seed">Seme per le operazioni random</param>
      public MachineLearningContext(int? seed = null) => _seed = seed;
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
      /// Verifica che un provider di contesto sia valido per l'ML.NET
      /// </summary>
      /// <param name="provider">Il provider</param>
      /// <param name="name">Nome del parametro</param>
      public static void AssertMLNET(IMachineLearningContextProvider provider, string name)
      {
         Contracts.AssertValue(provider, name);
         Contracts.AssertValue(provider.ML, $"{name}.{nameof(IMachineLearningContextProvider.ML)}");
         Contracts.AssertValue(provider.ML.NET, $"{name}.{nameof(IMachineLearningContextProvider.ML.NET)}");
      }
      /// <summary>
      /// Verifica che un provider di contesto sia valido per l'ML.NET
      /// </summary>
      /// <param name="provider">Il provider</param>
      /// <param name="name">Nome del parametro</param>
      public static void CheckMLNET(IMachineLearningContextProvider provider, string name)
      {
         Contracts.CheckValue(provider, name);
         Contracts.CheckValue(provider.ML, $"{name}.{nameof(IMachineLearningContextProvider.ML)}");
         Contracts.CheckValue(provider.ML.NET, $"{name}.{nameof(IMachineLearningContextProvider.ML.NET)}");
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
      #endregion
   }
}
