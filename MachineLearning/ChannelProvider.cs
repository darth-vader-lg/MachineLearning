using MachineLearning.Model;
using Microsoft.ML.Runtime;
using System;

namespace MachineLearning
{
   /// <summary>
   /// classe base per i provider di messaggi
   /// </summary>
   [Serializable]
   public abstract class ChannelProvider : ExceptionContext, IChannelProvider, IDisposable
   {
      #region Fields
      /// <summary>
      /// Canale di messaggistica
      /// </summary>
      private IChannel _channel;
      #endregion
      #region Properties
      /// <summary>
      /// Canale di messaggistica
      /// </summary>
      protected IChannel Channel => _channel ??= GetChannelProvider().Start((this as IModelName)?.ModelName ?? GetChannelProvider().ContextDescription);
      /// <summary>
      /// Indicatore di stato disposed
      /// </summary>
      [field: NonSerialized]
      public bool IsDisposed { get; private set; }
      #endregion
      #region Methods
      /// <summary>
      /// Forza il dispose dell'oggetto
      /// </summary>
      public void Dispose()
      {
         Dispose(disposing: true);
         GC.SuppressFinalize(this);
      }
      /// <summary>
      /// Funzione di dispose
      /// </summary>
      /// <param name="disposing">Se true indica che il dispose dell'oggetto e' stato chiamato da programma</param>
      protected virtual void Dispose(bool disposing)
      {
         if (!IsDisposed) {
            IsDisposed = true;
            if (disposing) {
               _channel?.Dispose();
               _channel = null;
            }
         }
      }
      /// <summary>
      /// Funzione di ottenimento del provider di canali
      /// </summary>
      /// <returns>Il provider</returns>
      protected abstract IChannelProvider GetChannelProvider();
      /// <summary>
      /// Funzione di ottenimento del contesto di eccezione
      /// </summary>
      /// <returns>Il contesto</returns>
      protected sealed override IExceptionContext GetExceptionContext() => GetChannelProvider();
      /// <summary>
      /// Crea un canale di messaggistica
      /// </summary>
      /// <param name="name">Nome del canale</param>
      /// <returns>Il canale</returns>
      IChannel IChannelProvider.Start(string name) => GetChannelProvider().Start(name);
      /// <summary>
      /// Crea una pipe di messaggistica
      /// </summary>
      /// <typeparam name="TMessage">Tipo di messaggio</typeparam>
      /// <param name="name">Nome della pipe</param>
      /// <returns>La pipe</returns>
      IPipe<TMessage> IChannelProvider.StartPipe<TMessage>(string name) => GetChannelProvider().StartPipe<TMessage>(name);
      #endregion
   }
}
