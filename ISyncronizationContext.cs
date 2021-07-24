using System;
using System.Threading;

namespace MachineLearning
{
   /// <summary>
   /// Interface for the syncronization contexts
   /// </summary>
   public interface ISyncronizationContext
   {
      #region Properties
      /// <summary>
      /// Indicate that it's needed a syncronized call due to the fact that we are not in the creation thread
      /// </summary>
      bool SyncRequired { get; }
      #endregion
      #region Methods
      /// <summary>
      /// Execute an action in the creation context of the machine learning context
      /// </summary>
      /// <param name="action">Action to execute</param>
      /// <param name="cancellation">Optional cancellatin token</param>
      void SyncExec(Action action, CancellationToken cancellation = default);
      /// <summary>
      /// Post an action to the creation context of the machine learning context
      /// </summary>
      /// <param name="action">Action to execute</param>
      void SyncPost(Action action);
      #endregion
   }
}
