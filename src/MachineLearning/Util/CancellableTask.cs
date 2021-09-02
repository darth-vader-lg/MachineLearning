using System;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace MachineLearning.Util
{
   /// <summary>
   /// Task cancellabile
   /// </summary>
   internal sealed class CancellableTask
   {
      #region Fields
      /// <summary>
      /// Sorgente del token di cancellazione
      /// </summary>
      private CancellationTokenSource cancellationTokenSource;
      #endregion
      #region Properties
      /// <summary>
      /// Token di cancellazione
      /// </summary>
      public CancellationToken CancellationToken => cancellationTokenSource.Token;
      /// <summary>
      /// Token di cancellazione passato alla creazione del task
      /// </summary>
      public CancellationToken ParentCancellationToken { get; private set; }
      /// <summary>
      /// Task associato
      /// </summary>
      public Task Task { get; private set; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      public CancellableTask() => StartNew(cancel => Task.CompletedTask);
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="task">Task da associare</param>
      public CancellableTask(Func<CancellationToken, Task> task) => StartNew(task);
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="task">Task da associare</param>
      /// <param name="cancellationToken">Token a cui connettere la cancellazione</param>
      public CancellableTask(Func<CancellationToken, Task> task, CancellationToken cancellationToken) => StartNew(task, cancellationToken);
      /// <summary>
      /// Funzione di cancellazione
      /// </summary>
      public void Cancel() =>cancellationTokenSource.Cancel();
      /// <summary>
      /// Restituisce l'awaiter
      /// </summary>
      /// <returns></returns>
      public TaskAwaiter GetAwaiter() => Task.GetAwaiter();
      /// <summary>
      /// Avvia un nuovo task
      /// </summary>
      /// <param name="task">Task da avviare</param>
      /// <returns>Il task avviato</returns>
      public Task StartNew(Func<CancellationToken, Task> task)
      {
         ParentCancellationToken = default;
         cancellationTokenSource = new CancellationTokenSource();
         return Task = task(CancellationToken);
      }
      /// <summary>
      /// Avvia un nuovo task
      /// </summary>
      /// <param name="cancellationToken">Token a cui connettere la cancellazione</param>
      /// <param name="task">Task da avviare</param>
      /// <returns>Il task avviato</returns>
      public Task StartNew(Func<CancellationToken, Task> task, CancellationToken cancellationToken)
      {
         cancellationTokenSource = CancellationTokenSource.CreateLinkedTokenSource(ParentCancellationToken = cancellationToken);
         return Task = task(CancellationToken);
      }
      #endregion
   }
}
