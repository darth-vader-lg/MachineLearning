using Microsoft.ML.Runtime;
using System;

namespace MachineLearning
{
   /// <summary>
   /// Classe base degli oggetti con provider di contesto
   /// </summary>
   /// <typeparam name="T">Tipo di contesto</typeparam>
   [Serializable]
   public abstract class ContextProvider<T> : ChannelProvider, IContextProvider<T> where T : class, IChannelProvider
   {
      #region Fields
      /// <summary>
      /// Provider di contesto
      /// </summary>
      private readonly IContextProvider<T> contextProvider;
      #endregion
      #region Properties
      /// <summary>
      /// Contesto
      /// </summary>
      public T Context => contextProvider.Context;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      public ContextProvider(IContextProvider<T> contextProvider)
      {
         MachineLearningContext.CheckContext(this.contextProvider = contextProvider, nameof(contextProvider));
         Contracts.CheckValue(contextProvider.Context, $"{nameof(contextProvider)}.{nameof(IContextProvider<T>.Context)}");
      }
      /// <summary>
      /// Funzione di restituzione del provider di canali
      /// </summary>
      /// <returns>Il provider</returns>
      protected sealed override IChannelProvider GetChannelProvider() => Context;
      #endregion
   }
}
